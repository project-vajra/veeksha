import os
from dataclasses import field

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

from veeksha.config.core.decorators import allow_from_file
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.constants.prefill_constants import PREFILL_POLYNOMIAL_DEGREE
from veeksha.logger import init_logger

logger = init_logger(__name__)


@allow_from_file
@frozen_dataclass
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
