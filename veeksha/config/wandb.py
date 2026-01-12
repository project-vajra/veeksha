from dataclasses import field
from typing import Optional

from veeksha.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass(allow_from_file=True)
class WandbConfig:
    """Configuration for Weights & Biases logging.

    Attributes:
        enabled: If True, the benchmark runner will initialize a wandb run and
            upload selected outputs/metrics.
        project: WandB project name. If unset, wandb will fall back to its
            standard resolution (e.g., `WANDB_PROJECT`).
        entity: WandB entity (user/team). Optional.
        group: WandB group name for logically-related runs (e.g., capacity search
            attempts or CLI sweeps). Optional.
        run_name: Optional run name. If unset, veeksha will default to the
            resolved output directory name for uniqueness.
        tags: Optional list of tags to attach to the run.
        notes: Optional human-readable notes.
        mode: Optional wandb mode override ("online", "offline", "disabled").
            If unset, wandb uses `WANDB_MODE` and its defaults.
        log_artifacts: If True, upload selected output files as a wandb Artifact.
    """

    enabled: bool = field(
        default=False, metadata={"help": "Enable Weights & Biases logging."}
    )
    project: Optional[str] = field(
        default=None,
        metadata={"help": "WandB project name (or set WANDB_PROJECT)."},
    )
    entity: Optional[str] = field(
        default=None,
        metadata={"help": "WandB entity (team/user). Optional."},
    )
    group: Optional[str] = field(
        default=None,
        metadata={"help": "WandB group name (for sweeps/capacity-search)."},
    )
    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "WandB run name. Defaults to the resolved output dir name."},
    )
    tags: list[str] = field(
        default_factory=list,
        metadata={"help": "List of WandB tags to attach to the run."},
    )
    notes: Optional[str] = field(
        default=None,
        metadata={"help": "Optional WandB notes for this run."},
    )
    mode: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional wandb mode override: 'online', 'offline', or 'disabled'."
        },
    )
    log_artifacts: bool = field(
        default=True,
        metadata={"help": "Upload selected output files as a wandb Artifact."},
    )
