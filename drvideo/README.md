# DrVideo with Qwen2.5-VL

A local implementation of [DrVideo: Document Retrieval Based Long Video Understanding](https://arxiv.org/abs/2406.12846).
One Qwen2.5-VL model plays every role the paper splits between LLaVA-NeXT and GPT.
It talks to any OpenAI-compatible API (vLLM, DashScope, LM Studio, Ollama's `/v1`, ...).

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your server's details:

```
BASE_URL=http://your-server:8000/v1
API_KEY=your-api-key
MODEL=Qwen/Qwen2.5-VL-7B-Instruct
```

`BASE_URL` is the part before `/chat/completions`, usually ending in `/v1`.
`MODEL` must be the exact model name your server reports at `{BASE_URL}/models`.
Environment variables with the same names override the `.env` file.
`.env` is git-ignored, so the key is never committed.

## Run

```bash
streamlit run app.py
```

To use it from other machines on the office network:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## How it maps to the paper

| Paper section | Code |
|---|---|
| 3.1 Video-document conversion: caption each sampled frame | `pipeline.build_document` |
| 3.2 Retrieval: top-K frames by cosine similarity to the question | `pipeline.retrieve` |
| 3.3 Augmentation: detailed caption (type A) or VQA (type B) for key frames | `pipeline.augment` |
| 3.4 Planning agent: is the document sufficient? | `pipeline.plan` |
| 3.4 Interaction agent: which frames need which information? | `pipeline.find_missing` |
| 3.5 Answering agent: chain-of-thought answer + confidence | `pipeline.answer` |
| Algorithm 1 | `pipeline.run` |

## Files

- `app.py`: Streamlit interface
- `pipeline.py`: the DrVideo method
- `prompts.py`: all prompts (adapted from the paper's appendix)
- `llm_client.py`: calls to the OpenAI-compatible API
- `video.py`: frame sampling
- `config.py`: connection settings and defaults (0.5 FPS, K = 5, 2 rounds, as in the paper)

## Notes

- Retrieval uses embeddings only if your server also hosts an embedding model and you set
  `EMBED_MODEL`. Otherwise it uses keyword (TF-IDF) similarity, which needs nothing extra.
- Captioning is the slow step (one API call per sampled frame). Captions are kept
  for the session, so follow-up questions on the same video are fast.
- The whole video document must fit in the model's context window. For long videos,
  lower the sampling rate or frame cap. Set `CONTEXT_TOKENS` to your server's limit
  so the app can warn you.
- If the server rejects JSON mode (`response_format`), the app switches it off
  automatically and relies on the prompts asking for JSON.
