"""All tunable settings in one place."""

OLLAMA_URL = "http://localhost:11434"

# One model plays every role: frame captioner, augmenter (VQA) and the three agents.
MODEL = "qwen2.5vl:7b"

# Embedding model for the retrieval module (`ollama pull nomic-embed-text`).
# If it is not installed, retrieval falls back to keyword (TF-IDF) similarity.
EMBED_MODEL = "nomic-embed-text"

# Context window used for every call. Keep it the same everywhere: Ollama
# reloads the model whenever it changes. Raise it for long videos if you have the VRAM.
NUM_CTX = 8192

# Paper defaults (Sec. 4.2 and the ablations).
SAMPLE_FPS = 0.5      # frames sampled per second of video
MAX_FRAMES = 90       # cap on document length; longer videos are sampled more sparsely
TOP_K = 5             # frames retrieved and augmented up front
MAX_ROUNDS = 2        # iterations of the planning/interaction loop

FRAME_SIZE = 512      # longest side of a frame sent to the model, in pixels
