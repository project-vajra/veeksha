from veeksha.config.microbenchmark import MicrobenchmarkConfig
from veeksha.logger import init_logger
from veeksha.microbenchmark.probe_registry import MicrobenchmarkProbeRegistry

logger = init_logger(__name__)


class Microbenchmark:
    def __init__(
        self,
        config: MicrobenchmarkConfig,
    ) -> None:
        self.config = config

    def _run_probe(self) -> None:
        """Run the configured probe (prefill/decode)."""
        probe_type = self.config.probe_config.get_type()
        logger.info("Running %s probe...", probe_type)

        runner = MicrobenchmarkProbeRegistry.get(
            probe_type,
            self.config,
        )

        runner.run()
        logger.info("%s probe completed.", probe_type)

    def run(self):
        """Run all enabled microbenchmark profilers."""
        logger.info(
            f"Starting microbenchmark profiling with output dir: {self.config.output_dir}"
        )

        self._run_probe()

        logger.info("Microbenchmark profiling completed!")
