"""Named benchmarks: a workload frozen behind a fingerprint.

A definition pins the exact request stream it generates, so a later run can
prove it is measuring the same workload. The pieces:

* :mod:`~veeksha.named_benchmark.define` — validate a definition, pin its
  workload fingerprint and provenance, optionally publish
* :mod:`~veeksha.named_benchmark.resolve` — load a definition at run time,
  apply free variables, and verify the pin
* :mod:`~veeksha.named_benchmark.knobs` — declared free variables (the only
  fields a run may change) and their CLI flags
* :mod:`~veeksha.named_benchmark.hub` — fetch and publish definitions on the
  Hugging Face Hub

The fingerprint itself lives in :mod:`veeksha.core.workload_fingerprint`, and
the environment it is pinned against in :mod:`veeksha.provenance`; both are
used outside named benchmarks.
"""

from veeksha.named_benchmark.resolve import (
    NamedBenchmarkError,
    check_workload_pin,
    resolve_named_benchmark,
)

__all__ = [
    "NamedBenchmarkError",
    "check_workload_pin",
    "resolve_named_benchmark",
]
