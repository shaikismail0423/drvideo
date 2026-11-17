"""Thin wrapper around the Ollama HTTP API."""

import base64
import json
import re

import requests

import config


def chat(prompt, image=None, as_json=False):
    """Send one prompt (optionally with a JPEG image) and return the reply text."""
    message = {"role": "user", "content": prompt}
    if image is not None:
        message["images"] = [base64.b64encode(image).decode()]

    body = {
        "model": config.MODEL,
        "messages": [message],
        "stream": False,
        "options": {"temperature": 0, "num_ctx": config.NUM_CTX},
    }
    if as_json:
        body["format"] = "json"

    response = requests.post(f"{config.OLLAMA_URL}/api/chat", json=body, timeout=600)
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def chat_json(prompt):
    """Like chat(), but parse the reply as JSON. Returns {} if the model's reply is unusable."""
    text = chat(prompt, as_json=True)
    match = re.search(r"\{.*\}", text, re.DOTALL)  # tolerate text around the JSON object
    try:
        reply = json.loads(match.group()) if match else {}
    except json.JSONDecodeError:
        reply = {}
    return reply if isinstance(reply, dict) else {}


def embed(texts):
    """Return one embedding per text, or None if the embedding model is unavailable."""
    try:
        response = requests.post(
            f"{config.OLLAMA_URL}/api/embed",
            json={"model": config.EMBED_MODEL, "input": texts},
            timeout=600,
        )
        response.raise_for_status()
        return response.json()["embeddings"]
    except (requests.RequestException, KeyError):
        return None
