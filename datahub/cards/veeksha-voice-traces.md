---
pretty_name: Veeksha Voice Traces
language:
  - en
task_categories:
  - automatic-speech-recognition
tags:
  - speech
  - benchmarking
  - asr
  - tts
  - veeksha
license: other
license_name: mixed
license_details: >-
  Per-pool licenses: VoxPopuli clips are CC0-1.0, Earnings22 clips are
  CC-BY-SA-4.0, AMI clips are CC-BY-4.0. See the Licensing section below.
configs:
  - config_name: aa_voxpopuli
    data_files: stt/aa_voxpopuli/manifest.jsonl
  - config_name: aa_earnings22
    data_files: stt/aa_earnings22/manifest.jsonl
  - config_name: ami_word_timed_2k
    data_files: stt/ami_word_timed_2k/manifest.jsonl
  - config_name: aa_bench100
    data_files: stt/aa_bench100/manifest.jsonl
  - config_name: aa_voxpopuli_tiled_5min
    data_files: stt/aa_voxpopuli_tiled_5min/manifest.jsonl
  - config_name: seed_tts_en
    data_files: tts/seed_tts_en/manifest.jsonl
  - config_name: sharegpt
    data_files: tts/sharegpt/manifest.jsonl
---

# Veeksha Voice Traces

Frozen, versioned audio trace bundles for benchmarking speech systems with
[Veeksha](https://github.com/project-vajra/veeksha), a high-fidelity
benchmarking framework for LLM and voice inference systems.

Rebuilding traces from upstream sources is not reproducible: upstream
datasets change revisions, transient fetch failures shift clip selection,
and forced alignment can jitter across GPUs. This repo freezes the traces
once so every benchmark run measures against identical references. **Cite
results as `repo @ tag` plus the manifest you used**, e.g.
`avartha/veeksha-voice-traces @ v0.1, stt/aa_bench100/manifest.jsonl`.

## Organization: pools, mixtures, derived datasets

```
stt/                          speech-to-text traces (audio in, text out)
  aa_voxpopuli/               POOL: source-pure, carries audio + provenance
  aa_earnings22/              POOL
  ami_word_timed_2k/          SUBSET: seeded slice of a corpus too large to
                              host in full (human-annotated word timings)
  aa_voxpopuli_tiled_5min/    DERIVED: transformed audio, own directory
  aa_bench100/                MIXTURE: manifest + recipe, no audio of its own
tts/                          text-to-speech traces (text in, audio out)
  seed_tts_en/                POOL: Seed-TTS-Eval English sentences (text rows)
  sharegpt/                   POOL: ShareGPT assistant turns (text rows +
                              native conversations for the sharegpt flavor)
```

- **Pools** are source-pure: one upstream dataset, one license, one
  provenance chain (`build_info.json`: builder git commit, CLI args,
  upstream revision SHAs, seed, aligner model/image).
- **Mixtures** compose pools (or other mixtures — composition is closed):
  a `manifest.jsonl` whose rows reference sibling pool audio
  (`../<pool>/audio/...`) plus a `mixture.json` recipe recording sources,
  counts, seed, and its transitively-flattened pool dependencies. Mixtures
  are a few KB; audio is never duplicated.
- **Subsets** are seeded samples of a pool, materializing their own copy of
  the audio they select so they stand alone (`build_info.json` records the
  parent, count, and seed, and inherits the parent's upstream revisions).
  Used when a corpus is too large to host in full; named for their size so
  they are not mistaken for the whole thing.
- **Derived datasets** contain transformed audio (e.g. tiling) and live in
  their own directory named `<parent>_<transform>`.

## Datasets

| Dataset | Kind | Clips | Word timestamps | Content |
|---|---|---|---|---|
| `stt/aa_voxpopuli` | pool | 631 | NeMo forced alignment | Read parliamentary speech (VoxPopuli), ≤30 s |
| `stt/aa_earnings22` | pool | 253 | NeMo forced alignment | Accented conversational earnings calls, ≤30 s |
| `stt/ami_word_timed_2k` | subset | 2,000 (2.8 h) | **Human annotations** | Spontaneous multi-party meeting speech (AMI), sampled across all 171 meetings |
| `stt/aa_bench100` | mixture | 100 (50+50) | inherited | The benchmark subset used in Veeksha's third-party comparisons |
| `stt/aa_voxpopuli_tiled_5min` | derived | 628 | NeMo forced alignment | VoxPopuli tiled to ~N(300 s, 30 s) for long-form streaming tests |
| `tts/seed_tts_en` | pool | 1,088 texts | — | Seed-TTS-Eval English sentences (frozen snapshot of `TwinkStart/Seed-TTS-Eval`) |
| `tts/sharegpt` | pool | 367,775 texts | — | ShareGPT assistant turns, flattened; native conversations included |

TTS pools are text-only (rows carry `text` instead of `audio_file`):
consume them with the `seed_tts_text` flavor (`local_path:` the manifest,
`id_column: sample_id`), or point the `sharegpt` flavor's `trace_file:` at
`tts/sharegpt/sharegpt_data.json` for its role/alpha-filter knobs. Text
mixtures compose with the same `mix` command and are fully self-contained.

AMI's timings come from the corpus's manual annotations — no aligner model
in the pipeline — making it the model-free anchor for word-latency metrics.
The AA pools' timestamps are model-derived (NeMo
`stt_en_fastconformer_hybrid_large_pc`); treat millisecond-level timing
comparisons across the two families accordingly.

`ami_word_timed_2k` is a seeded (seed 42) 2,000-clip sample of the full
68,291-clip / 97-hour AMI build, which is impractical to host as 68k
individual files. The sample spans all 171 meetings and all six recording
series (EN/ES/IB/IN/IS/TS), so it stays representative of the corpus's rooms
and accents; a size-ordered truncation would not. AMI is CC-BY-4.0 and
available upstream if you need the full corpus — rebuild it locally with
`prepare_audio_traces.py --datasets ami_word_timed`. Two source channels
(`ES2008a.A`, `TS3011d.B`) are skipped by that build because their
annotations lack word start/end times; this is a deterministic property of
the upstream data, so rebuilds skip exactly the same two.

## Manifest schema

One JSON object per line:

```json
{"session_id": 0,
 "audio_file": "audio/aa_voxpopuli/clip_00000.wav",
 "expected_transcript": "the committee will now hear...",
 "dataset": "aa_voxpopuli",
 "duration_s": 12.94,
 "sample_rate": 16000,
 "source_dataset": "ArtificialAnalysis/VoxPopuli-Cleaned-AA",
 "source_id": "20140311-0900-PLENARY-5-en_...",
 "sample_id": "20140311-0900-PLENARY-5-en_...",
 "reference_word_timestamps": [{"word": "the", "start_ms": 120.0, "end_ms": 180.0}]}
```

`audio_file` is relative to the manifest's directory (mixture rows point
into sibling pools). `sample_id` is the stable identity used for mixture
membership and deduplication.

## Using with Veeksha

```bash
# download the benchmark mixture; its pools are fetched automatically
python datahub/trace_hub.py fetch --datasets stt/aa_bench100 --revision v0.1
```

Then point your benchmark config's trace file at the manifest you want:

```yaml
session_generator:
  type: trace
  trace_file: traces/asr/aa_bench100/manifest.jsonl   # or a pool's manifest.jsonl
  wrap_mode: true
  flavor:
    type: audio
```

Compose your own benchmark sets without re-downloading or duplicating audio:

```bash
python datahub/trace_hub.py mix --name my_set \
    --take stt/aa_voxpopuli:200 --take stt/ami_word_timed_2k:200 --seed 7
```

With the same trace revision, manifest, and config `seed`, the generated
workload (clip selection, order, schedule) is bit-identical across machines.

## Using without Veeksha

```bash
hf download avartha/veeksha-voice-traces --repo-type dataset \
    --include "stt/aa_voxpopuli/*" --revision v0.1 --local-dir .
```

or via the `datasets` library (manifests only; audio stays file-referenced):

```python
from datasets import load_dataset
rows = load_dataset("avartha/veeksha-voice-traces", "aa_bench100", revision="v0.1")
```

## Licensing

Audio and transcripts are redistributed from their upstream sources under
their original licenses; each pool is license-pure:

- **`aa_voxpopuli`** (via `ArtificialAnalysis/VoxPopuli-Cleaned-AA`): CC0-1.0
- **`aa_earnings22`** (via `ArtificialAnalysis/Earnings22-Cleaned-AA`): CC-BY-SA-4.0
- **`ami_word_timed_2k`** (AMI Meeting Corpus): CC-BY-4.0

Mixtures inherit the licenses of the pools they draw from (recorded in
`mixture.json` → `requires`); subsets inherit their parent's (recorded in
`build_info.json` → `subset_of`). Upstream revision SHAs are pinned in each
pool's `build_info.json`.
