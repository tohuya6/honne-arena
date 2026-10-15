# Credits

- **Soham Padia**: [Steering Arena](https://sohampadianeu-steering-arena.hf.space/), the
  base of this project. The web UI's visual design, the steering scorer
  (`app/scoring.py`), the NDIF residual reader (`app/ndif_client.py`), the bounded
  scoring queue (`app/queue.py`), rate limiting, CAPTCHA, and their tests are his work,
  imported in the first commit with his authorship.
- **NDIF / NNsight** (ndif.us), for free remote access to model internals.
- Contests replicated in simplified form:
  - Aletheia's Quest (Cadenza Labs and NDIF, 2026), for the lie-detection track.
  - MIB, the Mechanistic Interpretability Benchmark (Mueller et al., ICML 2025), for
    the circuits track.
- Research the challenges build on:
  - Wang et al., "Interpretability in the Wild: a Circuit for Indirect Object
    Identification in GPT-2 small" (2022), for the IOI task and its known circuit.
  - Marks and Tegmark, "The Geometry of Truth" (2023), for the cities statement format.
- Models: GPT-2 small (OpenAI) and Qwen2.5-0.5B-Instruct (Qwen team), used through
  Hugging Face Transformers.
- Dynamic flag scoring follows [CTFd](https://github.com/CTFd/CTFd)'s formula.
