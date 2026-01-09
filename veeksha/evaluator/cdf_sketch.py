from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import rekha as rk
import wandb
from ddsketch import DDSketch

from veeksha.evaluator.plot_utils import (
    apply_axis_scale,
    format_axis_label,
    recommend_axis_scale,
)
from veeksha.logger import init_logger

logger = init_logger(__name__)


SUMMARY_PERCENTILES = [0.5, 0.9, 0.99]


class CDFSketch:
    def __init__(
        self,
        metric_name: str,
        should_write_to_wandb: bool = True,
        unit: Optional[str] = None,
    ) -> None:
        # metrics are a data series of two-dimensional (x, y) datapoints
        self.sketch = DDSketch(relative_accuracy=0.001)
        # column name
        self.metric_name = metric_name

        # most recently collected y datapoint for incremental updates
        # to aid incremental updates to y datapoints
        self.last_data = 0

        self.should_write_to_wandb = should_write_to_wandb
        self.unit = unit

    def __len__(self):
        return int(self.sketch.count)

    # add a new x, y datapoint
    def put(self, data: float) -> None:
        self.last_data = data
        self.sketch.add(data)

    def extend(self, values: List[float]):
        for value in values:
            self.put(value)

    # add a new datapoint as an incremental (delta) update to
    # recently collected datapoint
    def put_delta(self, delta: float) -> None:
        data = self.last_data + delta
        self.put(data)

    def print_distribution_stats(self, plot_name: str) -> None:
        if self.sketch._count == 0:
            return

        logger.info(
            f"{plot_name}: {self.metric_name} stats:"
            f" min: {self.sketch.get_quantile_value(0)},"
            f" max: {self.sketch.get_quantile_value(1)},"
            f" mean: {self.sketch.avg},"
            f" 25th percentile: {self.sketch.get_quantile_value(0.25)},"
            f" median: {self.sketch.get_quantile_value(0.5)},"
            f" 75th percentile: {self.sketch.get_quantile_value(0.75)},"
            f" 95th percentile: {self.sketch.get_quantile_value(0.95)},"
            f" 99th percentile: {self.sketch.get_quantile_value(0.99)}"
            f" 99.9th percentile: {self.sketch.get_quantile_value(0.999)}"
        )
        if wandb.run and self.should_write_to_wandb:
            wandb.log(
                {
                    f"{plot_name}_min": self.sketch.get_quantile_value(0),
                    f"{plot_name}_max": self.sketch.get_quantile_value(1),
                    f"{plot_name}_mean": self.sketch.avg,
                    f"{plot_name}_25th_percentile": self.sketch.get_quantile_value(
                        0.25
                    ),
                    f"{plot_name}_median": self.sketch.get_quantile_value(0.5),
                    f"{plot_name}_75th_percentile": self.sketch.get_quantile_value(
                        0.75
                    ),
                    f"{plot_name}_95th_percentile": self.sketch.get_quantile_value(
                        0.95
                    ),
                    f"{plot_name}_99th_percentile": self.sketch.get_quantile_value(
                        0.99
                    ),
                    f"{plot_name}_99.9th_percentile": self.sketch.get_quantile_value(
                        0.999
                    ),
                },
                step=0,
            )

    def _to_df(self) -> pd.DataFrame:
        # get quantiles at 1% intervals
        quantiles = np.linspace(0, 1, 101)
        # get quantile values (fall back to zeros when no samples were observed)
        if self.sketch.count == 0:
            quantile_values = [0.0 for _ in quantiles]
        else:
            quantile_values = [self.sketch.get_quantile_value(q) for q in quantiles]
        # create dataframe
        df = pd.DataFrame({"cdf": quantiles, self.metric_name: quantile_values})

        return df

    @property
    def sum(self) -> float:
        return self.sketch.sum

    def _save_df(self, df: pd.DataFrame, path: str, plot_name: str) -> None:
        df.to_csv(f"{path}/{plot_name}.csv")

        if wandb.run and self.should_write_to_wandb:
            wand_table = wandb.Table(dataframe=df)
            wandb.log({f"{plot_name}_table": wand_table}, step=0)

    def plot_cdf(
        self, path: str, plot_name: str, x_axis_label: Optional[str] = None
    ) -> None:
        if x_axis_label is None:
            x_axis_label = (
                f"{self.metric_name}{' (' + self.unit + ')' if self.unit else ''}"
            )

        raw_df = self._to_df()
        scale = recommend_axis_scale(raw_df[self.metric_name], kind="numeric")

        # Labels
        linear_x_label = x_axis_label
        scaled_x_label = (
            format_axis_label(self.metric_name, self.unit, scale)
            if scale != "linear"
            else linear_x_label
        )

        # Save linear plot always
        fig_linear = rk.line(
            raw_df,
            x=self.metric_name,
            y="cdf",
            markers=True,
            labels={self.metric_name: linear_x_label, "cdf": "CDF"},
        )
        fig_linear.save(f"{path}/{plot_name}.png", transparent=False)

        # If scaled (log/symlog), also save a scaled variant
        if scale != "linear":
            fig_scaled = rk.line(
                raw_df,
                x=self.metric_name,
                y="cdf",
                markers=True,
                labels={self.metric_name: scaled_x_label, "cdf": "CDF"},
            )
            apply_axis_scale(fig_scaled, axis="x", scale=scale)
            suffix = "log" if scale == "log" else "symlog"
            fig_scaled.save(f"{path}/{plot_name}_{suffix}_x.png", transparent=False)

        if wandb.run and self.should_write_to_wandb:
            wandb_df = raw_df.copy()
            wandb_df = wandb_df.rename(columns={self.metric_name: linear_x_label})

            wandb.log(
                {
                    f"{plot_name}_cdf": wandb.plot.line(
                        table=wandb.Table(dataframe=wandb_df),
                        x=linear_x_label,
                        y="cdf",
                        title=plot_name,
                    )
                },
                step=0,
            )

        self._save_df(raw_df, path, plot_name)

    def get_summary(self) -> Dict[str, Optional[float]]:
        return (
            {
                f"{self.metric_name} (Mean)": self.sketch.avg,
                **{
                    f"{self.metric_name} (P{int(p * 100)})": self.sketch.get_quantile_value(
                        p
                    )
                    for p in SUMMARY_PERCENTILES
                },
            }
            if self.sketch.count > 0
            else {
                f"{self.metric_name} (Mean)": 0,
                **{
                    f"{self.metric_name} (P{int(p * 100)})": 0
                    for p in SUMMARY_PERCENTILES
                },
            }
        )

    def to_csv_row(self) -> str:
        return ",".join([f"{v:.5f}" for v in self.get_summary().values()])

    def get_csv_header(self) -> str:
        return ",".join([f"{k}" for k in self.get_summary().keys()])

    def __str__(self) -> str:
        summary_str = ", ".join(
            [f"{k}: {v:.5f}" for k, v in self.get_summary().items()]
        )
        # remove the repeated metric name
        summary_str = summary_str.replace(self.metric_name, "")
        summary_str = summary_str.replace("(", "").replace(")", "").strip()
        # remove double spaces
        summary_str = " ".join(summary_str.split())
        return f"{self.metric_name} - {summary_str}"

    def __repr__(self) -> str:
        return self.__str__()
