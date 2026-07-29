# synthetic-concurrency

Sample **named benchmark** for Veeksha's define / pin / run workflow.

## What it freezes

- Synthetic single-request text sessions (`seed`, prompt length, output length)
- Tokenizer model (`gpt2` by default — needs `transformers` for define/run generation)
- Everything else in `config/base.yml`

## Free variable

| Knob | Target | Notes |
|------|--------|--------|
| `concurrency` | `traffic_scheduler.target_concurrent_sessions` | Load only; must not move the fingerprint |

## Commands

### Pin (define) — may use a normal (GIL) Python

Free-threaded 3.14 often lacks `tokenizers` wheels. Use a separate venv:

```bash
# one-time setup
uv venv .venv-define --python /usr/local/bin/python3.14   # non free-threaded 3.14+
uv pip install -e . --python .venv-define/bin/python

# pin (generation-only; no server)
# Session count comes from config.runtime.max_sessions in the definition.
.venv-define/bin/python -m veeksha benchmark define \
  --definition benchmarks/synthetic-concurrency
```

`benchmark define` is the only command allowed on a non–free-threaded interpreter.

### Run — free-threaded veeksha (`.venv314`)

```bash
.venv314/bin/python -m veeksha benchmark run \
  --benchmark benchmarks/synthetic-concurrency \
  --concurrency 4 \
  --endpoint.engine_type vllm \
  --endpoint.api_base http://localhost:8000/v1 \
  --endpoint.model gpt2 \
  --output_dir runs/synthetic-concurrency
```

With `runtime.pregenerate_sessions: true` (set in the base config), the run
checks the workload pin **before** any request is sent.
