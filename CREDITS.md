# Credits

- **Soham Padia**: [Steering Arena](https://sohampadianeu-steering-arena.hf.space/), the
  base of this project. The web UI and visual design, the steering scorer
  (`app/scoring.py`), the NDIF residual reader (`app/ndif_client.py`), the bounded
  scoring queue (`app/queue.py`), rate limiting, CAPTCHA, and their tests are his work,
  imported in the first commit with his authorship.
- **NDIF / NNsight** (ndif.us), for free remote access to model internals.
- Contest designs replicated in simplified form: Aletheia's Quest (Cadenza Labs and
  NDIF) and MIB, the Mechanistic Interpretability Benchmark (Mueller et al., ICML 2025).
- Dynamic flag scoring follows [CTFd](https://github.com/CTFd/CTFd)'s formula.
