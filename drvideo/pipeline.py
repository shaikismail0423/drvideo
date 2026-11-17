"""DrVideo: document-retrieval-based long video understanding (arXiv 2406.12846).

A "document" is a list with one entry per sampled frame:
    {"id": 1, "time": 0.0, "caption": "short description", "details": ["augmented info", ...]}

`frames` is the matching list of (timestamp, jpeg_bytes) from video.sample_frames;
frame id N is frames[N - 1].
"""

import copy
import re

import numpy as np

import prompts
from ollama_client import chat, chat_json, embed


# --- 3.1 Video-document conversion ------------------------------------------

def build_document(frames, on_progress=None):
    """Caption every frame to get the initial video document."""
    document = []
    for index, (time, image) in enumerate(frames):
        caption = chat(prompts.CAPTION, image=image)
        document.append({"id": index + 1, "time": time, "caption": caption, "details": []})
        if on_progress:
            on_progress(index + 1, len(frames))
    return document


def to_text(document):
    """Render the document the way the agents read it: one line per frame."""
    lines = []
    for entry in document:
        line = f"Frame {entry['id']}: {entry['caption']}"
        for detail in entry["details"]:
            line += f" [More detail: {detail}]"
        lines.append(line)
    return "\n".join(lines)


# --- 3.2 Document (frames) retrieval ----------------------------------------

def retrieve(document, question, k):
    """Return the ids of the k frames whose captions are most similar to the question."""
    texts = [question] + [entry["caption"] for entry in document]
    vectors = embed(texts)
    method = "embeddings"
    if vectors is None:
        vectors = _keyword_vectors(texts)
        method = "keywords (embedding model unavailable)"

    vectors = np.array(vectors, dtype=float)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    scores = vectors[1:] @ vectors[0]  # cosine similarity of each caption to the question
    top = np.argsort(scores)[::-1][:k]
    return sorted(int(i) + 1 for i in top), method


def _keyword_vectors(texts):
    """TF-IDF vectors: a dependency-free stand-in for an embedding model."""
    tokenized = [re.findall(r"[a-z0-9]+", text.lower()) for text in texts]
    vocabulary = {word: i for i, word in enumerate(sorted({w for words in tokenized for w in words}))}
    counts = np.zeros((len(texts), len(vocabulary)))
    for row, words in enumerate(tokenized):
        for word in words:
            counts[row, vocabulary[word]] += 1
    idf = np.log(len(texts) / (counts > 0).sum(axis=0)) + 1
    return counts * idf


# --- 3.3 Document augmentation ----------------------------------------------

def augment(document, frames, requests, question):
    """Add detailed information to the requested frames.

    `requests` is a list of (frame_id, type): "A" = detailed caption, "B" = answer the question.
    """
    for frame_id, kind in requests:
        prompt = prompts.DETAILED_CAPTION if kind == "A" else prompts.VQA.format(question=question)
        image = frames[frame_id - 1][1]
        detail = " ".join(chat(prompt, image=image).split())  # keep each frame on one line
        document[frame_id - 1]["details"].append(detail)


# --- 3.4 Multi-stage agent interaction loop ---------------------------------

def plan(document, question, memory):
    """Planning agent: is the document sufficient to answer? Returns (sufficient, explanation)."""
    reply = chat_json(prompts.PLANNING.format(**_header(document), question=question, memory=_memory_text(memory)))
    explanation = reply.get("explanation", "")
    if isinstance(explanation, list):
        explanation = " ".join(map(str, explanation))
    return str(reply.get("confidence")) == "1", explanation


def find_missing(document, question, memory, have, max_frames):
    """Interaction agent: which frames need which type of extra information?

    `have` maps "A"/"B" to the set of frame ids that already have that type.
    Returns a list of (frame_id, type).
    """
    reply = chat_json(prompts.INTERACTION.format(
        **_header(document), question=question, memory=_memory_text(memory), max_frames=max_frames,
        have_a=sorted(have["A"]) or "none", have_b=sorted(have["B"]) or "none",
    ))
    items = reply.get("frames")
    requests = []
    for item in items if isinstance(items, list) else []:
        try:
            frame_id, kind = int(item["frame"]), str(item["type"]).upper()
        except (KeyError, TypeError, ValueError):
            continue  # skip anything that is not {"frame": number, "type": letter}
        valid = 1 <= frame_id <= len(document) and kind in have and frame_id not in have[kind]
        if valid and (frame_id, kind) not in requests:
            requests.append((frame_id, kind))
    return requests[:max_frames]


# --- 3.5 Answering ----------------------------------------------------------

def answer(document, question, options=""):
    """Answering agent: chain-of-thought answer with a confidence score (1-3)."""
    choices = prompts.CHOICES.format(options=options) if options.strip() else ""
    return chat_json(prompts.ANSWERING.format(**_header(document), question=question, choices=choices))


# --- Algorithm 1: the whole thing -------------------------------------------

def run(initial_document, frames, question, options="", top_k=5, max_rounds=2):
    """Run DrVideo on an already-captioned video. Yields (step, data) as each step finishes."""
    document = copy.deepcopy(initial_document)  # keep the cached captions clean
    full_question = f"{question}\n{options}".strip()  # what the agents see
    have = {"A": set(), "B": set()}
    memory = []

    # Retrieve the top-K frames and augment them with the general (VQA) prompt.
    # Frames are asked the bare question: one frame cannot pick between answer
    # choices about the whole video, and its guess would pollute the document.
    top_frames, method = retrieve(document, question, top_k)
    augment(document, frames, [(i, "B") for i in top_frames], question)
    have["B"].update(top_frames)
    memory.append(f"Step 0: frames {top_frames} looked most related to the question and were given extra detail.")
    yield "retrieval", {"frames": top_frames, "method": method}

    # Planning agent <-> interaction agent loop.
    for round_number in range(1, max_rounds + 1):
        sufficient, explanation = plan(document, full_question, memory)
        yield "planning", {"round": round_number, "sufficient": sufficient, "explanation": explanation}
        if sufficient:
            break
        memory.append(f"Round {round_number}: judged insufficient. {explanation}")

        requests = find_missing(document, full_question, memory, have, top_k)
        yield "interaction", {"round": round_number, "requests": requests}
        if not requests:
            break
        augment(document, frames, requests, question)
        for frame_id, kind in requests:
            have[kind].add(frame_id)
        memory.append(f"Round {round_number}: extra detail was added to (frame, type) {requests}.")

    result = answer(document, question, options)
    yield "answer", {"result": result, "document": document}


def _header(document):
    seconds = int(document[-1]["time"])
    return {
        "duration": f"{seconds // 60} min {seconds % 60} s",
        "total": len(document),
        "document": to_text(document),
    }


def _memory_text(memory):
    return "\n".join(memory)
