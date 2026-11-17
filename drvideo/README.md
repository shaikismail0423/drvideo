# DrVideo with Ollama + Qwen2.5-VL

A local implementation of [DrVideo: Document Retrieval Based Long Video Understanding](https://arxiv.org/abs/2406.12846).
One model (`qwen2.5vl:7b`) plays every role the paper splits between LLaVA-NeXT and GPT.

## Run

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

Optional, for the paper's embedding-based retrieval (otherwise keyword similarity is used):

```bash
ollama pull nomic-embed-text
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
- `ollama_client.py`: calls to Ollama
- `video.py`: frame sampling
- `config.py`: model names and defaults (0.5 FPS, K = 5, 2 rounds, as in the paper)

## Notes

- Captioning is the slow step (about 4 s per frame on an RTX 4060). Captions are kept
  for the session, so follow-up questions on the same video are fast.
- The whole document must fit in the model's context (`NUM_CTX` in `config.py`).
  For long videos, lower the sampling rate or frame cap, or raise `NUM_CTX`.
