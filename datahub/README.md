# datahub

Tooling for building, versioning, and publishing benchmark datasets on the
Hugging Face Hub. Deliberately decoupled from the `veeksha` package: nothing
here imports veeksha, and veeksha only consumes the *outputs* (trace
directories under `traces/`) — so this directory can grow to serve other
dataset families (text traces, TTS corpora, ...) or be extracted into its
own repository without touching the benchmarker.

## Contents

```
datahub/
  trace_hub.py              generic Hub CLI: fetch / mix / subset / variant /
                            publish / card
  prepare_audio_traces.py   pool builder for the audio (STT) trace family
  prepare_text_traces.py    pool builder for the text (TTS) trace family
  cards/
    <repo-name>.md          one dataset card per Hub repo, uploaded as its README
```

## Concepts

- **One Hub repo per dataset family** (currently
  `avartha/veeksha-voice-traces`), organized by direction: `stt/<name>`,
  `tts/<name>`. Locally, `stt/<name>` maps to `traces/asr/<name>` — the
  layout veeksha sample configs expect.
- **Pools** are source-pure datasets that carry audio and `build_info.json`
  provenance (builder git commit, CLI args, upstream revision SHAs, seed,
  aligner model/image). One upstream source, one license, per pool.
- **Mixtures** compose pools and/or other mixtures: a manifest whose rows
  reference sibling pool audio plus a `mixture.json` recipe with
  transitively-flattened pool dependencies. Closed under composition, no
  audio duplication; `fetch` resolves a mixture's pool dependencies
  automatically.
- **Subsets** are seeded samples of a pool for corpora too large to host in
  full. Unlike a mixture, a subset materializes (hardlinks) the audio it
  selects, so it publishes without its parent; `build_info.json` records the
  parent, count, and seed and inherits the parent's upstream revisions. Name
  them for their size (`ami_word_timed_2k`) so they cannot be mistaken for
  the full corpus.
- **Derived datasets** (transformed audio) are sibling directories named
  `<parent>_<transform>`, built with `--output-name`.
- **Versioning**: immutable Hub tags. Tag after all of a release's uploads
  land; cite results as `repo @ tag` + manifest path. Repos are created
  private by default (`--public` to opt out).

## Typical flows

```bash
# consumer: pull the benchmark mixture at a pinned version
python datahub/trace_hub.py fetch --datasets stt/aa_bench100 --revision v0.1

# maintainer: build pools, compose, publish, document
python datahub/prepare_audio_traces.py --datasets aa_voxpopuli --clips-per-dataset 0
python datahub/trace_hub.py mix --name aa_bench100 \
    --take stt/aa_voxpopuli:50 --take stt/aa_earnings22:50 --seed 42
python datahub/trace_hub.py publish --datasets stt/aa_voxpopuli,stt/aa_bench100 --tag v0.1
python datahub/trace_hub.py card
```

See `.claude/skills/voice-traces/SKILL.md` for the full workflow rules.
