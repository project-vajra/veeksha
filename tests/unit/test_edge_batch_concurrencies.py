import pytest

from veeksha.sweeps.utils import GRAPH_LADDER_EDGES, edge_batch_concurrencies


@pytest.mark.unit
def test_default_edges_straddle_the_graph_ladder_boundaries() -> None:
    values = edge_batch_concurrencies()
    # Sorted, de-duplicated, and straddling each ladder edge by +/- 1.
    assert list(values) == sorted(set(values))
    assert values == (383, 384, 385, 447, 448, 449, 479, 480, 481, 511, 512, 513)
    # The batch-boundary "+1" concurrencies from the canary spec are present.
    for edge_plus_one in (385, 449, 481, 513):
        assert edge_plus_one in values


@pytest.mark.unit
def test_straddle_width_is_configurable() -> None:
    assert edge_batch_concurrencies([512], straddle=2) == (510, 511, 512, 513, 514)
    assert edge_batch_concurrencies([512], straddle=0) == (512,)


@pytest.mark.unit
def test_minimum_clamps_out_low_values() -> None:
    assert edge_batch_concurrencies([2], straddle=3, minimum=1) == (1, 2, 3, 4, 5)


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(edges=[]), "edges must be non-empty"),
        (dict(edges=[0]), "edge values must be >= 1"),
        (dict(edges=[512], straddle=-1), "straddle must be >= 0"),
        (dict(edges=[512], minimum=0), "minimum must be >= 1"),
    ],
)
def test_edge_batch_concurrencies_rejects_invalid(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        edge_batch_concurrencies(**kwargs)


@pytest.mark.unit
def test_graph_ladder_edges_default() -> None:
    assert GRAPH_LADDER_EDGES == (384, 448, 480, 512)
