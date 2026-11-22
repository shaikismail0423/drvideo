"""Calls to an OpenAI-compatible API (vLLM, DashScope, LM Studio, Ollama's /v1, ...)."""

import base64
import json
import re

import requests

import config


class APIError(Exception):
    """The server could not be reached or returned something unusable."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status  # HTTP status code, if the server answered


_json_mode_supported = True  # switched off if the server rejects response_format


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
        except APIError as error:
            if error.status not in (400, 422):
                raise
            _json_mode_supported = False  # the prompts still ask for JSON, so retry without it

    return _reply_text(_post("chat/completions", body))


def chat_json(prompt):
    """Like chat(), but parse the reply as a JSON object. Returns {} if the reply is unusable."""
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
    except (APIError, KeyError, TypeError):
        return None


def _post(path, body):
    url = f"{config.BASE_URL}/{path}"
    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {config.API_KEY}"},
            json=body,
            timeout=config.TIMEOUT,
        )
    except requests.RequestException as error:
        raise APIError(f"Could not reach {url}: {error}") from error

    if not response.ok:
        raise APIError(f"{url} returned {response.status_code}: {response.text[:500]}", response.status_code)
    try:
        return response.json()
    except ValueError as error:
        raise APIError(f"{url} did not return JSON: {response.text[:200]}") from error


def _reply_text(response):
    try:
        return (response["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as error:
        raise APIError(f"Unexpected response from the server: {str(response)[:300]}") from error
