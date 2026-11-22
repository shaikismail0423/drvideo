"""All tunable settings in one place.

Connection details are read from environment variables, or from a `.env` file
next to this one (see `.env.example`). Never put the API key in this file.
"""

import os
from pathlib import Path


def _load_env_file():
    """Read KEY=VALUE lines from .env without overriding real environment variables."""
    path = Path(__file__).with_name(".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.strip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_env_file()

# OpenAI-compatible endpoint serving Qwen2.5-VL (vLLM, DashScope, LM Studio, Ollama's /v1, ...).
BASE_URL = os.getenv("BASE_URL", "").rstrip("/")   # e.g. http://your-server:8000/v1
API_KEY = os.getenv("API_KEY", "")                # can be empty if the server needs no key

# One model plays every role: frame captioner, augmenter (VQA) and the three agents.
MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

# Optional embedding model on the same server, for the retrieval module.
# Leave empty to use keyword (TF-IDF) similarity instead.
EMBED_MODEL = os.getenv("EMBED_MODEL", "")

# Rough size of the model's context window, in tokens. Only used to warn when the
# video document is likely too long; the real limit is set on the server.
CONTEXT_TOKENS = int(os.getenv("CONTEXT_TOKENS") or 32000)

MAX_TOKENS = 1024     # longest reply allowed from the model
TIMEOUT = 300         # seconds to wait for one reply

# Paper defaults (Sec. 4.2 and the ablations).
SAMPLE_FPS = 0.5      # frames sampled per second of video
MAX_FRAMES = 90       # cap on document length; longer videos are sampled more sparsely
TOP_K = 5             # frames retrieved and augmented up front
MAX_ROUNDS = 2        # iterations of the planning/interaction loop

FRAME_SIZE = 512      # longest side of a frame sent to the model, in pixels
