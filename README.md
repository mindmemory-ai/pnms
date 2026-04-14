# PNMS — Personal Neural Memory System

Most chat apps hit the same wall: the model only sees what fits in the window. PNMS is a small Python library that sits beside your LLM and keeps a **per-user** memory you can grow over time—facts, habits, preferences—without stuffing everything into every prompt.

What you get in one place:

- **Slot memory**: vectors plus text, retrieved by similarity, updated or merged when new content is close enough to something old.
- **A co-occurrence graph**: when slots show up together, edges get stronger; at query time you can pull in one-hop neighbors so related memories surface together.
- **Optional concept modules**: tiny networks that learn to re-rank slots inside a cluster when you turn that on.
- **Checkpoints on disk**: save and load so restarts don’t wipe the user’s world.

PNMS does **not** call OpenAI or Ollama for you. You pass a function `(query, context) -> reply`; PNMS builds `context` from memory and hands you the rest.

## What’s in this repo

| Path | Role |
|------|------|
| `pnms/` | Installable package |
| `docs/` | Design notes (`pnms.md`) and public API notes (`pnms_api.md`) |
| `examples/` | Scripts to exercise cold start, graph, concepts, save/load |

The `paper/` directory is for local LaTeX drafts and is ignored by git (see `.gitignore`).

## Install

Python 3.9+ recommended.

```bash
cd pnms
pip install -r requirements.txt
pip install -e .
```

**Dependencies (roughly):** PyTorch (vectors and concept nets), sentence-transformers (optional but typical for good embeddings), scikit-learn (clustering when forming concepts), SQLite via the standard library for `graph.db`.

## A minimal run

```python
from pnms import (
    PNMS,
    PNMSConfig,
    SimpleQueryEncoder,
    setup_basic_logging,
    PNMSClient,
    HandleQueryResult,
)

setup_basic_logging("INFO")

config = PNMSConfig(
    concept_enabled=True,
    graph_enabled=True,
    concept_checkpoint_dir="data/pnms_ckpt",
)

encoder = SimpleQueryEncoder(embed_dim=config.embed_dim or 64, vocab_size=10000)
engine = PNMS(config=config, user_id="user_1", encoder=encoder)

def my_llm(prompt: str) -> str:
    return "Sure—here’s a short answer."

answer = engine.handle_query(
    query="What language do I like for algorithms?",
    llm=lambda q, ctx: my_llm(ctx),
    content_to_remember="Prefers Python for algorithm work.",
    system_prompt="Answer using the context when it helps.",
)
print(answer)

client = PNMSClient(config)
result: HandleQueryResult = client.handle(
    user_id="user_1",
    query="What language do I like for algorithms?",
    llm=lambda q, ctx: my_llm(ctx),
    content_to_remember="Prefers Python for algorithm work.",
    system_prompt="Answer using the context when it helps.",
)
print(result.response)
```

Read-only context (no writes this round):

```python
context, num_slots = engine.get_context_for_query(
    query="Quadratic formula?",
    system_prompt="You are a helpful assistant.",
    use_concept=True,
)
```

## Logging and errors

The library uses the standard `logging` module. Namespaces you’ll see often: `pnms.system`, `pnms.memory`, `pnms.graph`, `pnms.concept`.

Exceptions inherit from `PNMSError`—for example `ConfigError`, `PersistenceError`, `LLMError`. Catch what you need and degrade gracefully (skip concepts, retry disk, show a plain reply, etc.).

## Versions worth knowing

There are **two** version lines:

- **Library version** — what you installed (`pnms.__version__` / `LIBRARY_VERSION`). Bump this when you ship code changes.
- **Memory format version** — what your checkpoint files expect (`MEMORY_FORMAT_VERSION`). Bump this when the on-disk layout changes.

They can move at different speeds: you might upgrade the package without touching the file format. See `docs/pnms_api.md` for `get_versions()`, `peek_checkpoint_versions()`, and load-time checks.

Current defaults: library `0.1.0`, memory format `1.0.0` (check `pnms/versioning.py` if you need the exact strings).

## Examples folder

Under `examples/` you’ll find scripts for cold start, graph behavior, concepts, and a longer save/load sanity check (`full_flow_save_load_verify.py`). After you change core logic, running a few of these is a good smoke test against `docs/pnms.md`.

---

If something in the docs and the code disagrees, trust the code path you’re actually running—and open an issue so we can align them.
