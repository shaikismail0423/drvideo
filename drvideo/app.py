"""Streamlit interface for DrVideo. Run with: streamlit run app.py"""

import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st

import config
import pipeline
from llm_client import APIError
from video import sample_frames


def load_video(upload, fps, max_frames):
    """Sample and caption the video once; later questions on it reuse the result."""
    data = upload.getvalue()
    key = (hashlib.md5(data).hexdigest(), fps, max_frames)
    if st.session_state.get("video_key") == key:
        return st.session_state["frames"], st.session_state["document"]

    # OpenCV needs a file on disk; keep the original extension so it picks the right decoder.
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.name).suffix) as file:
        file.write(data)
    try:
        frames = sample_frames(file.name, fps, max_frames)
    finally:
        os.remove(file.name)
    if not frames:
        st.error("Could not read any frames from this video.")
        st.stop()

    bar = st.progress(0.0, "Captioning frames...")
    document = pipeline.build_document(
        frames, lambda done, total: bar.progress(done / total, f"Captioning frame {done}/{total}")
    )
    bar.empty()
    st.session_state.update(video_key=key, frames=frames, document=document)
    return frames, document


def show_step(step, data):
    """Print one step of the agent loop as it happens."""
    if step == "retrieval":
        st.write(f"**Retrieval** ({data['method']}): augmented frames {data['frames']}")
    elif step == "planning":
        verdict = "sufficient" if data["sufficient"] else "insufficient"
        st.write(f"**Planning agent, round {data['round']}**: {verdict}. {data['explanation']}")
    elif step == "interaction":
        requested = ", ".join(f"frame {i} (type {kind})" for i, kind in data["requests"]) or "nothing"
        st.write(f"**Interaction agent, round {data['round']}**: requested {requested}")


def show_answer(result, document, frames):
    """Show the final answer and the key frames that were augmented."""
    st.subheader("Answer")
    st.success(str(result.get("final_answer") or "The model did not return an answer."))
    st.write(f"**Confidence:** {result.get('confidence', '?')} / 3")
    st.write(f"**Explanation:** {result.get('explanation', '')}")
    with st.expander("Step-by-step reasoning"):
        st.write(result.get("reasoning", ""))

    key_frames = [entry for entry in document if entry["details"]]
    st.subheader(f"Key frames ({len(key_frames)})")
    for entry in key_frames:
        image_column, text_column = st.columns([1, 3])
        image_column.image(frames[entry["id"] - 1][1], caption=f"Frame {entry['id']} ({entry['time']:.0f}s)")
        text_column.write(f"**Caption:** {entry['caption']}")
        for detail in entry["details"]:
            text_column.write(f"**Added detail:** {detail}")

    with st.expander("Final video document"):
        st.text(pipeline.to_text(document))


def answer_question(upload, question, options, settings):
    frames, document = load_video(upload, settings["fps"], settings["max_frames"])
    st.write(f"Video document: {len(document)} frames.")
    if len(pipeline.to_text(document)) / 4 > config.CONTEXT_TOKENS * 0.8:  # ~4 characters per token
        st.warning("The document may not fit in the model's context window. "
                   "Lower the sampling rate or the maximum number of frames.")

    with st.status("Running DrVideo...", expanded=True) as status:
        for step, data in pipeline.run(document, frames, question, options,
                                       settings["top_k"], settings["max_rounds"]):
            show_step(step, data)
        status.update(label="Done", state="complete")

    show_answer(data["result"], data["document"], frames)  # the last step is always "answer"


# --- Page --------------------------------------------------------------------

st.set_page_config(page_title="DrVideo", page_icon="🎬", layout="wide")
st.title("DrVideo")
st.caption(f"Document-retrieval-based long video question answering with {config.MODEL}.")

if not config.BASE_URL:
    st.error("BASE_URL is not set. Copy .env.example to .env and fill it in "
             "(or set environment variables), then restart the app.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    settings = {
        "fps": st.number_input("Sampling rate (frames per second)", 0.05, 2.0, config.SAMPLE_FPS, 0.05),
        "max_frames": st.number_input("Maximum frames", 5, 500, config.MAX_FRAMES, 5),
        "top_k": st.number_input("Top-K retrieved frames", 1, 20, config.TOP_K),
        "max_rounds": st.number_input("Maximum agent rounds", 0, 5, config.MAX_ROUNDS),
    }

upload = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv", "webm"])
question = st.text_input("Question")
options = st.text_area("Answer choices (optional, one per line)", placeholder="A. ...\nB. ...", height=100)
if upload:
    st.video(upload)

if st.button("Answer", type="primary", disabled=not (upload and question.strip())):
    try:
        answer_question(upload, question.strip(), options, settings)
    except APIError as error:
        st.error(f"The model API call failed. Check BASE_URL, API_KEY and MODEL in .env.\n\n{error}")
