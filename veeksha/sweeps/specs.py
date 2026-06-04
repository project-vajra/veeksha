"""Static sweep specifications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

CONCURRENCY_SWEEP = "concurrency"
INPUT_SWEEP = "input"

MODEL_ALIASES = {
    "qwen-tts": "qwen-tts",
    "qwen3-tts": "qwen-tts",
    "tts": "qwen-tts",
    "qwen3-omni": "qwen3-omni",
    "omni": "qwen3-omni",
    "higgs-audio-v2": "higgs-audio-v2",
    "higgs-audio": "higgs-audio-v2",
    "higgs": "higgs-audio-v2",
    "fish-speech": "fish-speech",
    "fish-speech-s2-pro": "fish-speech",
    "fishaudio-s2-pro": "fish-speech",
    "s2-pro": "fish-speech",
    "vibe-voice": "vibe-voice",
    "vibe": "vibe-voice",
}


@dataclass(frozen=True)
class SweepSpec:
    sweep_type: str
    engine: str
    model: str
    config_name: str
    temp_prefix: str
    run_config_template: str
    run_name_template: str
    default_concurrencies: Tuple[int, ...] = ()
    default_concurrency: Optional[int] = None
    default_range_start: Optional[int] = None
    default_range_end: Optional[int] = None
    default_step: Optional[int] = None
    write_runtime_limits: bool = True
    disable_audio_for_input: bool = False

    @property
    def config_path(self) -> Path:
        return CONFIG_DIR / self.config_name


SPECS: Dict[Tuple[str, str, str], SweepSpec] = {
    (CONCURRENCY_SWEEP, "vajra", "qwen-tts"): SweepSpec(
        sweep_type=CONCURRENCY_SWEEP,
        engine="vajra",
        model="qwen-tts",
        config_name="vajra.yaml",
        temp_prefix="vajra_qwen_tts_sweep",
        run_config_template="vajra_qwen_c{concurrency}.yaml",
        run_name_template="vajra_qwen3tts_c_{concurrency}_10_minutes",
        default_concurrencies=(1, 2, 4, 8, 16, 32, 64),
    ),
    (CONCURRENCY_SWEEP, "vllm", "qwen-tts"): SweepSpec(
        sweep_type=CONCURRENCY_SWEEP,
        engine="vllm",
        model="qwen-tts",
        config_name="tts_vllm_omni.yaml",
        temp_prefix="tts_vllm_omni_sweep",
        run_config_template="tts_vllm_omni_c{concurrency}.yaml",
        run_name_template="tts_vllm_omni_c_{concurrency}_10_minutes",
        default_concurrencies=(1, 2, 4, 8, 16, 32, 64, 128),
    ),
    (CONCURRENCY_SWEEP, "sglang", "qwen-tts"): SweepSpec(
        sweep_type=CONCURRENCY_SWEEP,
        engine="sglang",
        model="qwen-tts",
        config_name="tts_sglang_omni.yaml",
        temp_prefix="tts_sglang_omni_sweep",
        run_config_template="tts_sglang_omni_c{concurrency}.yaml",
        run_name_template="tts_sglang_omni_c_{concurrency}_10_minutes",
        default_concurrencies=(1, 2, 4, 8, 16, 32, 64, 128),
    ),
    (CONCURRENCY_SWEEP, "vllm", "higgs-audio-v2"): SweepSpec(
        sweep_type=CONCURRENCY_SWEEP,
        engine="vllm",
        model="higgs-audio-v2",
        config_name="higgs_audio_v2_vllm_omni.yaml",
        temp_prefix="higgs_audio_v2_vllm_omni_sweep",
        run_config_template="higgs_audio_v2_vllm_omni_c{concurrency}.yaml",
        run_name_template="higgs_audio_v2_vllm_omni_c_{concurrency}_10_minutes",
        default_concurrencies=(1, 2, 4, 8, 16, 32, 64, 128),
    ),
    (CONCURRENCY_SWEEP, "vllm", "fish-speech"): SweepSpec(
        sweep_type=CONCURRENCY_SWEEP,
        engine="vllm",
        model="fish-speech",
        config_name="fish_speech_vllm_omni.yaml",
        temp_prefix="fish_speech_vllm_omni_sweep",
        run_config_template="fish_speech_vllm_omni_c{concurrency}.yaml",
        run_name_template="fish_speech_vllm_omni_c_{concurrency}_10_minutes",
        default_concurrencies=(1, 2, 4, 8, 16, 32, 64, 128),
    ),
    (CONCURRENCY_SWEEP, "vajra", "fish-speech"): SweepSpec(
        sweep_type=CONCURRENCY_SWEEP,
        engine="vajra",
        model="fish-speech",
        config_name="vajra_fish_speech_s2_pro.yaml",
        temp_prefix="vajra_fish_speech_s2_pro_sweep",
        run_config_template="vajra_fish_speech_s2_pro_c{concurrency}.yaml",
        run_name_template="vajra_fish_speech_s2_pro_c_{concurrency}_10_minutes",
        default_concurrencies=(1, 2, 4, 8, 16, 32, 64, 128),
    ),
    (CONCURRENCY_SWEEP, "vajra", "higgs-audio-v2"): SweepSpec(
        sweep_type=CONCURRENCY_SWEEP,
        engine="vajra",
        model="higgs-audio-v2",
        config_name="vajra_higgs_audio_v2.yaml",
        temp_prefix="vajra_higgs_audio_v2_sweep",
        run_config_template="vajra_higgs_audio_v2_c{concurrency}.yaml",
        run_name_template="vajra_higgs_audio_v2_c_{concurrency}_10_minutes",
        default_concurrencies=(1, 2, 4, 8, 16, 32, 64, 128),
    ),
    (CONCURRENCY_SWEEP, "vajra", "qwen3-omni"): SweepSpec(
        sweep_type=CONCURRENCY_SWEEP,
        engine="vajra",
        model="qwen3-omni",
        config_name="vajra_qwen.yaml",
        temp_prefix="vajra_qwen3_omni_sweep",
        run_config_template="vajra_qwen3_omni_c{concurrency}.yaml",
        run_name_template="vajra_qwen3_omni_c_{concurrency}_10_minutes",
        default_concurrencies=(1, 2, 4, 8, 16, 32, 64, 128),
    ),
    (CONCURRENCY_SWEEP, "vllm", "qwen3-omni"): SweepSpec(
        sweep_type=CONCURRENCY_SWEEP,
        engine="vllm",
        model="qwen3-omni",
        config_name="qwen3_omni.yaml",
        temp_prefix="vllm_qwen3_omni_sweep",
        run_config_template="vllm_qwen3_omni_c{concurrency}.yaml",
        run_name_template="vllm_qwen3_omni_c_{concurrency}_10_minutes",
        default_concurrencies=(1, 2, 4, 8, 16, 32, 64, 128),
    ),
    (CONCURRENCY_SWEEP, "vajra", "vibe-voice"): SweepSpec(
        sweep_type=CONCURRENCY_SWEEP,
        engine="vajra",
        model="vibe-voice",
        config_name="vajra_vibe_voice.yaml",
        temp_prefix="vajra_vibe_voice_0_5_sweep",
        run_config_template="vajra_vibe_voice_c{concurrency}.yaml",
        run_name_template="vj_vibe_voice_0.5_{date_tag}_c={concurrency}",
        default_concurrencies=(1, 2, 4, 8),
        write_runtime_limits=False,
    ),
    (INPUT_SWEEP, "vajra", "qwen-tts"): SweepSpec(
        sweep_type=INPUT_SWEEP,
        engine="vajra",
        model="qwen-tts",
        config_name="vajra.yaml",
        temp_prefix="vajra_qwen_tts_inputsweep",
        run_config_template="vajra_qwen_c{concurrency}_chars{input_size}.yaml",
        run_name_template=(
            "vajra_qwen3tts_c_{concurrency}_chars_{input_size}_10_minutes"
        ),
        default_concurrency=64,
        default_range_start=380,
        default_range_end=500,
        default_step=40,
        disable_audio_for_input=True,
    ),
    (INPUT_SWEEP, "vllm", "qwen-tts"): SweepSpec(
        sweep_type=INPUT_SWEEP,
        engine="vllm",
        model="qwen-tts",
        config_name="tts_vllm_omni.yaml",
        temp_prefix="vllm_qwen_tts_inputsweep",
        run_config_template="vllm_qwen_tts_c{concurrency}_chars{input_size}.yaml",
        run_name_template=(
            "vllm_qwen3tts_c_{concurrency}_chars_{input_size}_10_minutes"
        ),
        default_concurrency=16,
        default_range_start=180,
        default_range_end=500,
        default_step=40,
        disable_audio_for_input=True,
    ),
    (INPUT_SWEEP, "vllm", "higgs-audio-v2"): SweepSpec(
        sweep_type=INPUT_SWEEP,
        engine="vllm",
        model="higgs-audio-v2",
        config_name="higgs_audio_v2_vllm_omni.yaml",
        temp_prefix="vllm_higgs_audio_v2_inputsweep",
        run_config_template="vllm_higgs_audio_v2_c{concurrency}_chars{input_size}.yaml",
        run_name_template=(
            "vllm_higgs_audio_v2_c_{concurrency}_chars_{input_size}_10_minutes"
        ),
        default_concurrency=16,
        default_range_start=180,
        default_range_end=500,
        default_step=40,
        disable_audio_for_input=True,
    ),
    (INPUT_SWEEP, "vllm", "qwen3-omni"): SweepSpec(
        sweep_type=INPUT_SWEEP,
        engine="vllm",
        model="qwen3-omni",
        config_name="qwen3_omni.yaml",
        temp_prefix="vllm_qwen3_omni_inputsweep",
        run_config_template="vllm_qwen3_omni_c{concurrency}_chars{input_size}.yaml",
        run_name_template=(
            "vllm_qwen3_omni_c_{concurrency}_chars_{input_size}_10_minutes"
        ),
        default_concurrency=16,
        default_range_start=20,
        default_range_end=500,
        default_step=40,
        disable_audio_for_input=True,
    ),
}


def supported_combinations() -> str:
    rows = sorted(SPECS)
    return "\n".join(
        f"  {kind:11s} {engine:8s} {model}" for kind, engine, model in rows
    )
