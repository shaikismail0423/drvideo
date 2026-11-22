"""Thin wrapper around an OpenAI-compatible API (POST {BASE_URL}/chat/completions)."""

import base64
import json
import re

import requests

import config

_json_mode_supported = True  # switched off if the server rejects response_format


def _post(path, body):
    response = requests.post(
        f"{config.BASE_URL}/{path}",
        headers={"Authorization": f"Bearer {config.API_KEY}"},
        json=body,
        timeout=config.TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def chat(prompt, image=None, as_json=False):
    """Send one prompt (optionally with a JPEG image) and return the reply text."""
    global _json_mode_supported

    content = prompt
    if image is not None:
        data_url = "data:image/jpeg;base64," + base64.b64encode(image).decode()
        content = [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": prompt},
        ]

    body = {
        "model": config.MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": config.MAX_TOKENS,
    }
    if as_json and _json_mode_supported:
        try:
            return _reply_text(_post("chat/completions", {**body, "response_format": {"type": "json_object"}}))
        except requests.HTTPError as error:
            if error.response.status_code != 400:
                raise
            _json_mode_supported = False  # the prompt still asks for JSON, so fall through

    return _reply_text(_post("chat/completions", body))


def chat_json(prompt):
    """Like chat(), but parse the reply as JSON. Returns {} if the model's reply is unusable."""
    text = chat(prompt, as_json=True)
    match = re.search(r"\{.*\}", text, re.DOTALL)  # tolerate text or ``` fences around the JSON
    try:
        reply = json.loads(match.group()) if match else {}
    except json.JSONDecodeError:
        reply = {}
    return reply if isinstance(reply, dict) else {}


def embed(texts):
    """Return one embedding per text, or None if no embedding model is configured or it fails."""
    if not config.EMBED_MODEL:
        return None
    try:
        data = _post("embeddings", {"model": config.EMBED_MODEL, "input": texts})["data"]
        return [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]
    except (requests.RequestException, KeyError, TypeError):
        return None


def _reply_text(response):
    return (response["choices"][0]["message"]["content"] or "").strip()
