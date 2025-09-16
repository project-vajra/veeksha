from functools import partial
from typing import List

from veeksha.lm_eval.api.filter import FilterEnsemble
from veeksha.lm_eval.api.registry import get_filter

# trigger registration side-effects
from veeksha.lm_eval.filters import custom as _custom  # noqa: F401
from veeksha.lm_eval.filters import decontamination as _decontamination  # noqa: F401
from veeksha.lm_eval.filters import extraction as _extraction  # noqa: F401
from veeksha.lm_eval.filters import selection as _selection  # noqa: F401


def build_filter_ensemble(
    filter_name: str, components: List[List[str]]
) -> FilterEnsemble:
    """
    Create a filtering pipeline.
    """
    filters = []
    for function, kwargs in components:
        if kwargs is None:
            kwargs = {}
        # create a filter given its name in the registry
        f = partial(get_filter(function), **kwargs)
        # add the filter as a pipeline step
        filters.append(f)

    return FilterEnsemble(name=filter_name, filters=filters)
