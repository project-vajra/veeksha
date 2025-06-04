import json
import os
import re
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

from veeksha.config.base_poly_config import BasePolyConfig
from veeksha.config.flat_dataclass import create_flat_dataclass, get_config_class_by_type_name
from veeksha.config.utils import dataclass_to_dict
from veeksha.constants.prefill_constants import PREFILL_POLYNOMIAL_DEGREE
from veeksha.core.llm_clients import SUPPORTED_APIS
from veeksha.logger import init_logger
from veeksha.types import (
    RequestGeneratorType,
    RequestIntervalGeneratorType,
    RequestLengthGeneratorType,
)

logger = init_logger(__name__)


@dataclass
class BaseRequestIntervalGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )


@dataclass
class TraceRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/AzureFunctionsInvocationTraceForTwoWeeksJan2021Processed.csv",
        metadata={"help": "Path to the trace file for request intervals."},
    )
    start_time: str = field(
        default="1970-01-01 00:00:00", metadata={"help": "Start time for the trace."}
    )
    end_time: str = field(
        default="1970-01-01 04:00:00", metadata={"help": "End time for the trace."}
    )
    time_scale_factor: float = field(
        default=0.3,
        metadata={"help": "Factor to scale the time intervals in the trace."},
    )

    @classmethod
    def get_type(cls):
        return RequestIntervalGeneratorType.TRACE


@dataclass
class PoissonRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = field(
        default=1.0,
        metadata={"help": "Queries per second for the Poisson distribution."},
    )

    @classmethod
    def get_type(cls):
        return RequestIntervalGeneratorType.POISSON


@dataclass
class GammaRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = field(
        default=1.0, metadata={"help": "Queries per second for the Gamma distribution."}
    )
    cv: float = field(
        default=0.5,
        metadata={"help": "Coefficient of variation for the Gamma distribution."},
    )

    @classmethod
    def get_type(cls):
        return RequestIntervalGeneratorType.GAMMA


@dataclass
class StaticRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    @classmethod
    def get_type(cls):
        return RequestIntervalGeneratorType.STATIC


@dataclass
class BaseRequestLengthGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42, metadata={"help": "Random seed for the request length generator."}
    )
    max_tokens: int = field(
        default=4096, metadata={"help": "Maximum number of tokens allowed."}
    )


@dataclass
class TraceRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv",
        metadata={"help": "Path to the trace file for request lengths."},
    )
    prefill_scale_factor: float = field(
        default=1, metadata={"help": "Scale factor for prefill tokens."}
    )
    decode_scale_factor: float = field(
        default=1, metadata={"help": "Scale factor for decode tokens."}
    )

    block_size: int = field(
        default=512, metadata={"help": "Number of tokens per block."}
    )

    @classmethod
    def get_type(cls) -> RequestLengthGeneratorType:
        return RequestLengthGeneratorType.TRACE


@dataclass
class ZipfRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    theta: float = field(
        default=0.6, metadata={"help": "Theta parameter for the Zipf distribution."}
    )
    scramble: bool = field(
        default=False, metadata={"help": "Whether to scramble the Zipf distribution."}
    )
    min_tokens: int = field(
        default=1024, metadata={"help": "Minimum number of tokens."}
    )
    prefill_to_decode_ratio: float = field(
        default=20.0, metadata={"help": "Ratio of prefill tokens to decode tokens."}
    )

    @classmethod
    def get_type(cls) -> RequestLengthGeneratorType:
        return RequestLengthGeneratorType.ZIPF


@dataclass
class UniformRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    min_tokens: int = field(
        default=1024, metadata={"help": "Minimum number of tokens."}
    )
    prefill_to_decode_ratio: float = field(
        default=20.0, metadata={"help": "Ratio of prefill tokens to decode tokens."}
    )

    @classmethod
    def get_type(cls) -> RequestLengthGeneratorType:
        return RequestLengthGeneratorType.UNIFORM


@dataclass
class FixedRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    prefill_tokens: int = field(
        default=4096, metadata={"help": "Number of prefill tokens."}
    )
    decode_tokens: int = field(
        default=512, metadata={"help": "Number of decode tokens."}
    )

    @classmethod
    def get_type(cls) -> RequestLengthGeneratorType:
        return RequestLengthGeneratorType.FIXED


@dataclass
class BaseRequestGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42, metadata={"help": "Random seed for the request generator."}
    )
    max_tokens: int = field(
        default=8192, metadata={"help": "Maximum number of tokens allowed."}
    )

    def __post_init__(self):
        self.length_generator_config: BaseRequestLengthGeneratorConfig = None  # type: ignore


@dataclass
class SyntheticRequestGeneratorConfig(BaseRequestGeneratorConfig):
    @classmethod
    def get_type(cls):
        return RequestGeneratorType.SYNTHETIC


@dataclass
class PrefixRequestGeneratorConfig(BaseRequestGeneratorConfig):
    @classmethod
    def get_type(cls):
        return RequestGeneratorType.PREFIX


@dataclass
class TraceRequestGeneratorConfig(BaseRequestGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/sydney_enterprise.csv",
        metadata={"help": "Path to the trace file for request generation."},
    )
    date: str = field(
        default="2023-08-21", metadata={"help": "Date for the trace data."}
    )
    prefill_scale_factor: float = field(
        default=0.3, metadata={"help": "Scale factor for prefill tokens."}
    )
    decode_scale_factor: float = field(
        default=1, metadata={"help": "Scale factor for decode tokens."}
    )
    time_scale_factor: float = field(
        default=0.04, metadata={"help": "Scale factor for time intervals."}
    )

    @classmethod
    def get_type(cls):
        return RequestGeneratorType.TRACE


@dataclass
class LmevalRequestGeneratorConfig(BaseRequestGeneratorConfig):
    tasks: list = field(
        default_factory=lambda: [],
        metadata={"help": "The tasks to evaluate the language model on."},
    )
    num_fewshot: int = field(
        default=1,
        metadata={"help": "The number of fewshot examples to use for the tasks."},
    )
    limit: int = field(
        default=10,
        metadata={"help": "The number of examples to evaluate on."},
    )
    is_logit_based: bool = field(
        default=False,
        metadata={
            "help": "Whether the evaluation is logit based. If True, the task will be evaluated using OpenAI Completions API."
        },
    )

    @classmethod
    def get_type(cls):
        return RequestGeneratorType.LMEVAL


@dataclass
class ClientConfig:
    model: str = field(
        default="gpt-3.5-turbo",
        metadata={"help": "The model to use for this load test."},
    )
    tokenizer: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer to use for this load test. By default, the tokenizer is inferred from the model."
        },
    )
    num_clients: int = field(
        default=2,
        metadata={"help": "The number of clients to use for benchmark."},
    )
    num_concurrent_requests_per_client: int = field(
        default=5,
        metadata={"help": "The number of concurrent requests to send per client."},
    )
    additional_sampling_params: str = field(
        default="{}",
        metadata={
            "help": "Additional sampling params to send with the each request to the LLM API. "
            "By default, no additional sampling params are sent."
        },
    )
    llm_api: str = field(
        default="openai_chat",
        metadata={
            "help": f"The name of the llm api to use. Can select from {SUPPORTED_APIS}"
        },
    )
    address_append_value: str = field(
        default="chat/completions",
        metadata={"help": "The address append value for OpenAI API."},
    )

    def __post_init__(self):
        self.additional_sampling_params_dict = {}

        if self.additional_sampling_params:
            self.additional_sampling_params_dict = json.loads(
                self.additional_sampling_params
            )

        if self.tokenizer is None:
            self.tokenizer = self.model


@dataclass
class MetricsConfig:
    output_dir: str = field(
        default="benchmark_results",
        metadata={"help": "The directory to save the benchmark results to."},
    )
    should_use_given_dir: bool = field(
        default=True,
        metadata={
            "help": "Whether to add directly use output_dir directory or create new directories for the results."
        },
    )
    should_write_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to write metrics to wandb."},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "The wandb project to log metrics to."},
    )
    wandb_group: Optional[str] = field(
        default=None,
        metadata={"help": "The wandb group to log metrics to."},
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "The wandb run name to log metrics to."},
    )
    enable_wandb_sweep: bool = field(
        default=False,
        metadata={"help": "Whether to enable wandb sweep."},
    )
    wandb_sweep_id: Optional[str] = field(
        default=None,
        metadata={"help": "The wandb sweep id to log metrics to."},
    )
    wandb_sweep_name: Optional[str] = field(
        default=None,
        metadata={"help": "The wandb sweep name to log metrics to."},
    )


@dataclass
class DeadlineConfig:
    ttft_deadline: float = field(
        default=0.1,
        metadata={"help": "The deadline for time to first token."},
    )
    tbt_deadline: float = field(
        default=0.05,
        metadata={"help": "The deadline between tokens."},
    )
    target_deadline_miss_rate: float = field(
        default=0.1,
        metadata={"help": "The target deadline miss rate."},
    )
    ttft_slack: float = field(
        default=0.0,
        metadata={
            "help": "The slack for time to first token. Only used if use_predictions_for_ttft is True."
        },
    )


@dataclass
class PrefillProfilerConfig:
    prefill_lengths: list = field(
        default_factory=lambda: [],
        metadata={"help": "The lengths to prefill the profiler with."},
    )
    cache_predictions: bool = field(
        default=True,
        metadata={"help": "Whether to cache the predictions for the prefill profiler."},
    )
    use_predictions_for_ttft: bool = field(
        default=False,
        metadata={"help": "Whether to use the predictions from the prefill profiler."},
    )
    max_prefill_tokens_to_predict: int = field(
        default=int(2**20),
        metadata={
            "help": "The maximum number of tokens to predict for the prefill profiler."
        },
    )
    predictor_dir: str = field(
        default="",
        metadata={"help": "The path to directory of prefill predictor."},
    )

    def do_predictions(self, start_token_count=1):
        model_path = os.path.join(self.predictor_dir, "prefill_predictor.pkl")

        if not os.path.exists(model_path):
            logger.error(f"Predictor not found at {model_path}. Exiting.")
            return

        self.predictions = {}

        model: RandomForestRegressor = joblib.load(model_path)
        transformer = PolynomialFeatures(
            degree=PREFILL_POLYNOMIAL_DEGREE, include_bias=False
        )
        x = np.arange(
            start=start_token_count, stop=self.max_prefill_tokens_to_predict + 1
        ).reshape(-1, 1)
        x_poly = transformer.fit_transform(x)
        y = model.predict(x_poly)
        for i in range(len(x)):
            self.predictions[int(x[i][0])] = y[i]

    def save_predictions(self):
        """Save the predictions to a file to same directory for future use."""
        predictions_path = os.path.join(self.predictor_dir, "prefill_predictions.pkl")
        joblib.dump(self.predictions, predictions_path)

    def __post_init__(self):
        self.predictions = None

    def fill_predictions_array(self):
        assert (
            self.use_predictions_for_ttft
        ), "Predictions should be used for TTFT to fill predictions array."
        assert (
            self.predictor_dir
        ), "Predictor path must be provided if use_predictions is True."
        predictions_path = os.path.join(self.predictor_dir, "prefill_predictions.pkl")
        logger.info(f"Getting prefill predictions from path: {predictions_path}")
        if os.path.exists(predictions_path):
            self.predictions = joblib.load(predictions_path)
            if len(self.predictions) < self.max_prefill_tokens_to_predict:
                logger.warning(
                    f"Predictions found at {predictions_path} but not enough predictions found. Loading predictor and predicting more tokens."
                )
                self.do_predictions()
                self.save_predictions()
        else:
            logger.warning(
                f"Predictions not found at {predictions_path}. Loading predictor and predicting."
            )
            self.do_predictions()
            self.save_predictions()


@dataclass
class BenchmarkConfig(ABC):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )
    timeout: int = field(
        default=1200,
        metadata={"help": "The amount of time to run the load test for."},
    )
    max_completed_requests: int = field(
        default=10,
        metadata={
            "help": "The number of requests to complete before finishing the test. Note "
            "that its possible for the test to timeout first."
        },
    )
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        metadata={"help": "The time stamp for the benchmark."},
    )
    client_config: ClientConfig = field(
        default_factory=ClientConfig,
        metadata={"help": "The client configuration for the benchmark."},
    )
    metrics_config: MetricsConfig = field(
        default_factory=MetricsConfig,
        metadata={"help": "The metrics configuration for the benchmark."},
    )
    deadline_config: DeadlineConfig = field(
        default_factory=DeadlineConfig,
        metadata={"help": "The deadline configuration for the benchmark."},
    )
    prefill_profiler_config: PrefillProfilerConfig = field(
        default_factory=PrefillProfilerConfig,
        metadata={"help": "The prefill profiler configuration for the benchmark."},
    )
    request_interval_generator_config: BaseRequestIntervalGeneratorConfig = field(
        default_factory=TraceRequestIntervalGeneratorConfig,
        metadata={
            "help": "The request interval generator configuration for the benchmark."
        },
    )
    request_length_generator_config: BaseRequestLengthGeneratorConfig = field(
        default_factory=TraceRequestLengthGeneratorConfig,
        metadata={
            "help": "The request length generator configuration for the benchmark."
        },
    )
    request_generator_config: BaseRequestGeneratorConfig = field(
        default_factory=SyntheticRequestGeneratorConfig,
        metadata={"help": "The request generator configuration for the benchmark."},
    )

    def __post_init__(self):
        if not os.path.exists(self.metrics_config.output_dir):
            os.makedirs(self.metrics_config.output_dir)

        if not self.metrics_config.should_use_given_dir:
            benchmark_identifier = f"{self.client_config.model}_{self.request_interval_generator_config.get_type()}_{self.request_generator_config.get_type()}"
            benchmark_identifier = re.sub(r"[^\w\d-]+", "-", benchmark_identifier)
            benchmark_identifier = re.sub(r"-{2,}", "-", benchmark_identifier)

            self.metrics_config.output_dir = os.path.join(
                self.metrics_config.output_dir, benchmark_identifier, self.timestamp
            )

        if self.prefill_profiler_config.use_predictions_for_ttft:
            self.prefill_profiler_config.max_prefill_tokens_to_predict = max(
                self.prefill_profiler_config.max_prefill_tokens_to_predict,
                self.request_generator_config.max_tokens,
            )
            self.prefill_profiler_config.fill_predictions_array()

        # assign the length generator config to the request generator config
        self.request_generator_config.length_generator_config = (
            self.request_length_generator_config
        )
        self.request_length_generator_config.max_tokens = (
            self.request_generator_config.max_tokens
        )

        if self.request_generator_config.get_type() == RequestGeneratorType.LMEVAL:
            logger.warning("Removing timeout for LMEval.")
            self.timeout = -1
            assert isinstance(
                self.request_generator_config, LmevalRequestGeneratorConfig
            )

            if self.request_generator_config.is_logit_based:
                self.client_config.llm_api = "openai_completions"
                self.client_config.address_append_value = "completions"
            else:
                self.client_config.llm_api = "openai_chat"
                self.client_config.address_append_value = "chat/completions"

        self.write_config_to_file()

    @classmethod
    def create_from_cli_args(cls):
        flat_config = create_flat_dataclass(cls).create_from_cli_args()
        instance = flat_config.reconstruct_original_dataclass()
        instance.__flat_config__ = flat_config
        return instance

    def to_dict(self):
        if not hasattr(self, "__flat_config__"):
            logger.warning("Flat config not found. Returning the original config.")
            return self.__dict__

        return self.__flat_config__.__dict__  # type: ignore

    def write_config_to_file(self):
        config_dict = dataclass_to_dict(self)
        with open(
            os.path.join(f"{self.metrics_config.output_dir}", "config.json"), "w"
        ) as f:
            json.dump(config_dict, f, indent=4)

    @classmethod
    def generate_capacity_search_benchmark_configs(cls, configs: dict):
        all_benchmarks = []
        
        benchmark_configs = configs.get("benchmark", [])
        client_configs = configs.get("client", [])
        request_generator_configs = configs.get("request_generator", [])
        request_interval_generator_configs = configs.get("request_interval_generator", [])
        request_length_generator_configs = configs.get("request_length_generator", [])
        
        if not request_interval_generator_configs:
            request_interval_generator_configs = [{}]
            
        if not request_length_generator_configs:
            request_length_generator_configs = [{}]
        
        for (
            benchmark_config,
            client_config,
            request_generator_config,
            request_interval_generator_config,
            request_length_generator_config,
        ) in product(
            benchmark_configs,
            client_configs,
            request_generator_configs,
            request_interval_generator_configs,
            request_length_generator_configs,
        ):
            interval_config_dict = dict(request_interval_generator_config)
            interval_type = interval_config_dict.pop("type")
            
            interval_config_class = get_config_class_by_type_name(
                BaseRequestIntervalGeneratorConfig,
                interval_type
            )
            request_interval_generator_config = interval_config_class(**interval_config_dict)

            length_config_dict = dict(request_length_generator_config)
            length_type = length_config_dict.pop("type")
            
            length_config_class = get_config_class_by_type_name(
                BaseRequestLengthGeneratorConfig,
                length_type
            )
            request_length_generator_config = length_config_class(**length_config_dict)
            
            gen_config_dict = dict(request_generator_config)
            gen_type = gen_config_dict.pop("type")
            
            gen_config_class = get_config_class_by_type_name(
                BaseRequestGeneratorConfig,
                gen_type
            )
            request_generator_config = gen_config_class(**gen_config_dict)
            
            client_config = ClientConfig(**client_config)

            # TODO: we don't want to pass manually set values to the benchmark config
            # we should pass all available ones
            benchmark_config = cls(
                **benchmark_config,
                client_config=client_config,
                request_generator_config=request_generator_config,
                request_interval_generator_config=request_interval_generator_config,
                request_length_generator_config=request_length_generator_config,
            )

            all_benchmarks.append(benchmark_config)

        return all_benchmarks

@dataclass
class CapacitySearchConfig:
    """Configuration for capacity search benchmark. This is a special benchmark that runs multiple benchmarks with different QPS and
    finds the maximum QPS that can be sustained given the deadline constraints."""

    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator for capacity search."},
    )
    start_qps: float = field(
        default=1,
        metadata={"help": "The starting QPS for the capacity search."},
    )
    num_qps_steps: int = field(
        default=10,
        metadata={"help": "The number of QPS steps for the capacity search."},
    )
    min_search_granularity: float = field(
        default=2.5,
        metadata={"help": "Minimum search granularity for capacity (%)"},
    )
    max_iterations: int = field(
        default=20,
        metadata={"help": "Maximum number of iterations for capacity search."},
    )
    output_dir: str = field(
        default="./veeksha/capacity_search/output",
        metadata={"help": "Output directory for capacity search."},
    )
    benchmark_config_file: str = field(
        default="./veeksha/capacity_search/config/default_config.yml",
        metadata={"help": "Path to benchmark config file."},
    )
    server_config_file: Optional[str] = field(
        # TODO: make the optional flow work
        default=None,
        metadata={"help": "Path to server launch command file"},
    )
    slo_type: str = field(
        default="deadline",
        metadata={"help": "Type of SLO to use for capacity search"},
    )
    tbt_slo: float = field(
        default=0.03,
        metadata={"help": "TBT SLO for capacity search"},
    )
    tbt_percentile: float = field(
        default=0.99,
        metadata={"help": "TBT percentile for capacity search"},
    )
    ttft_slo: float = field(
        default=0.1,
        metadata={"help": "TTFT SLO for capacity search"},
    )
    ttft_percentile: float = field(
        default=0.9,
        metadata={"help": "TTFT percentile for capacity search"},
    )
    tpot_slo: float = field(
        default=0.1,
        metadata={"help": "TPOT SLO for capacity search"},
    )
    tpot_percentile: float = field(
        default=0.9,
        metadata={"help": "TPOT percentile for capacity search"},
    )
    ttft_slack_slo: float = field(
        default=0.3,
        metadata={"help": "TTFT slack SLO for capacity search"},
    )
    deadline_miss_rate_slo: float = field(
        default=0.1,
        metadata={"help": "Deadline miss rate SLO for capacity search"},
    )
    deadline_miss_rate_percentile: float = field(
        default=0.99,
        metadata={"help": "Deadline miss rate percentile for capacity search"},
    )
    dynamic_ttft_slo: bool = field(
        default=True,
        metadata={"help": "Dynamic TTFT SLO for capacity search"},
    )
    # TODO: remove from arg, move to trace config or similar
    trace_session_match_threshold: float = field(
        default=0.9,
        metadata={"help": "Trace session match threshold for capacity search"},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb project for capacity search"},
    )
    enable_wandb_sweep: bool = field(
        default=False,
        metadata={"help": "Enable wandb sweep for capacity search"},
    )
    wandb_sweep_name: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb sweep name for capacity search"},
    )
    wandb_sweep_id: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb sweep id for capacity search"},
    )

    @classmethod
    def create_from_cli_args(cls):
        flat_config = create_flat_dataclass(cls).create_from_cli_args()
        instance = flat_config.reconstruct_original_dataclass()
        instance.__flat_config__ = flat_config
        return instance

    def to_dict(self):
        if not hasattr(self, "__flat_config__"):
            logger.warning("Flat config not found. Returning the original config.")
            return self.__dict__

        return self.__flat_config__.__dict__  # type: ignore

    def write_config_to_file(self):
        config_dict = dataclass_to_dict(self)
        with open(
            os.path.join(f"{self.output_dir}", "config.json"), "w"
        ) as f:
            json.dump(config_dict, f, indent=4)
