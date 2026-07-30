---
name: voice-traces
description: Fetch, build, mix, and publish veeksha voice trace datasets on the Hugging Face Hub (avartha/veeksha-voice-traces)
triggers:
  - voice traces
  - audio traces
  - trace datasets
  - fetch traces
  - publish traces
  - dataset variant
  - dataset mixture
  - hf dataset
---

# Voice Traces

Use this skill for anything involving veeksha's voice trace datasets: the
Hugging Face repo `avartha/veeksha-voice-traces`, local `traces/` builds,
mixtures/variants, and the publish/tag workflow. Tooling lives in
`datahub/` (see `datahub/README.md`), decoupled from the veeksha package.

## Key concepts

- **Hub repo layout**: `stt/<name>` and `tts/<name>` directories, one per
  dataset. The local checkout maps `stt/<name>` to `traces/asr/<name>` (the
  path sample configs expect). Tools: `datahub/trace_hub.py`
  (fetch/mix/variant/publish/card) and `datahub/prepare_audio_traces.py`
  (pool builds).
- **Pool**: source-pure dataset carrying audio + `build_info.json`
  provenance (git commit, argv, upstream revision SHAs, seed, aligner).
  One upstream source, one license. Built by `prepare_audio_traces.py`;
  a single-source build lands at `traces/asr/<dataset>` automatically.
- **Mixture**: composes pools and/or other mixtures. Own directory with
  `manifest.jsonl` (rows reference sibling pool audio via
  `../<pool>/audio/...`) + `mixture.json` (recipe + `requires`). Mixing is
  closed under composition: mixtures can mix mixtures. `requires` is always
  flattened to pools; dedupe by `sample_id` (opt out with
  `--allow-duplicates` for intentional oversampling). No audio duplication.
- **Subset**: seeded sample of a pool for corpora too large to host in full.
  Materializes (hardlinks) its own audio, so unlike a mixture it publishes
  without its parent; `build_info.json` records parent/count/seed and
  inherits upstream revisions. Name it for its size (`ami_word_timed_2k`).
- **Derived dataset**: transformed audio (tiling, augmentation). Own
  directory named `<parent>_<transform>`, built with `--output-name`.
- **Versioning**: immutable HF tags (`v0.1`, ...). Cite results as
  `repo @ tag` + manifest path. Tag when a version will be cited.

## Common tasks

Fetch for benchmarking (mixture dependencies are pulled automatically):

    python datahub/trace_hub.py fetch --datasets stt/aa_bench100 --revision v0.1

Build a pool from upstream sources (one build per source):

    python datahub/prepare_audio_traces.py --datasets aa_voxpopuli --clips-per-dataset 0
    python datahub/prepare_audio_traces.py --datasets ami_word_timed --clips-per-dataset 0

Build TTS text pools (rows carry `text` instead of `audio_file`; consume via
the seed_tts_text flavor's `local_path` + `id_column: sample_id`, or the
sharegpt flavor's `trace_file:` at the pool's native sharegpt_data.json):

    python datahub/prepare_text_traces.py --datasets seed_tts_en
    python datahub/prepare_text_traces.py --datasets sharegpt \
        --sharegpt-source /path/to/sharegpt_data.json

Build a derived dataset (own directory, never clobbers pools):

    python datahub/prepare_audio_traces.py --datasets aa_voxpopuli \
        --target-duration 300 --clips-per-dataset 0 \
        --output-name aa_voxpopuli_tiled_5min

Compose a mixture (seeded sampling, or reference-manifest membership):

    python datahub/trace_hub.py mix --name my_set \
        --take stt/aa_voxpopuli:50 --take stt/aa_earnings22:50 --seed 42
    python datahub/trace_hub.py mix --name aa_bench100 \
        --reference path/to/reference_manifest.jsonl \
        --sources stt/aa_voxpopuli,stt/aa_earnings22

Slice a too-large corpus into a standalone, publishable pool:

    python datahub/trace_hub.py subset --dataset stt/ami_word_timed \
        --name ami_word_timed_2k --count 2000 --seed 42

Publish (validates first, excludes alignment/ scratch; repos are created
private by default, pass --public to opt out; tag last):

    python datahub/trace_hub.py publish \
        --datasets stt/aa_voxpopuli,stt/aa_bench100 --tag v0.2
    python datahub/trace_hub.py card   # upload datahub/cards/<repo-name>.md

## Rules

1. **Validate before publishing**: `publish` refuses broken dirs, but check
   build logs for `WARNING: skipping` lines — skipped source rows shift clip
   selection and hurt reproducibility. A clean build has zero.
2. **Pools stay source-pure**: never build multiple upstream datasets into
   one directory; compose with `mix` instead.
3. **Keep published datasets under ~10k files** (HF's per-directory
   guidance; existing pools are ~600). For a bigger corpus, publish a
   `subset` and name it for its size — don't reach for
   `upload_large_folder`. Subset with `subset` (seeded across the whole
   corpus), never `prepare_audio_traces.py --clips-per-dataset`, which
   truncates iteration order: on AMI that yields 4 of 171 meetings from a
   single recording series.
4. **Never publish `alignment/`** (NeMo intermediates) — the publish command
   excludes it automatically; don't work around that.
5. **New dataset or mixture ⇒ update the dataset card**
   (`datahub/cards/veeksha-voice-traces.md`): add the `configs:` entry and
   table row, then run the `card` subcommand. Configs must only reference
   files that exist in the repo, or the HF viewer breaks.
6. **Tag AFTER all uploads** for a release land, so one tag covers every
   dataset consistently.
7. **Word-timestamp trust levels differ**: AA pools use NeMo forced
   alignment (model-derived); `ami_word_timed_2k` uses human annotations
   (model-free anchor). Don't mix them in fine-grained timing comparisons
   without noting this.
8. HF auth: `hf auth login` in a real terminal (interactive prompt fails in
   non-TTY sessions). Verify with `hf auth whoami` — needs write access to
   `avartha`.
