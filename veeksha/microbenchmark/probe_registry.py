from veeksha.microbenchmark.decode_probe import DecodeProbe
from veeksha.microbenchmark.prefill_probe import PrefillProbe
from veeksha.types import MicrobenchmarkProbeType
from veeksha.types.base_registry import BaseRegistry


class MicrobenchmarkProbeRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> MicrobenchmarkProbeType:  # type: ignore
        return MicrobenchmarkProbeType.from_str(key_str)  # type: ignore


MicrobenchmarkProbeRegistry.register(MicrobenchmarkProbeType.PREFILL, PrefillProbe)
MicrobenchmarkProbeRegistry.register(MicrobenchmarkProbeType.DECODE, DecodeProbe)
