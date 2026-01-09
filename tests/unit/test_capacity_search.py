"""Unit tests for capacity search logic."""

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from veeksha.capacity_search import (
    _adaptive_capacity_search,
    _knob_description,
    patch_traffic_knob,
    run_capacity_search,
)
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.capacity_search import CapacitySearchConfig
from veeksha.config.generator.interval import FixedIntervalGeneratorConfig
from veeksha.config.traffic import ConcurrentTrafficConfig, RateTrafficConfig


@pytest.mark.unit
def test_adaptive_search_binary_logic_integer() -> None:
    """Test the adaptive search logic with integer domain (e.g. concurrency)."""
    
    # Passing: x <= 10. Max: 20. Start: 2. Expansion: 2.
    # Phase 1: 2(P) -> 4(P) -> 8(P) -> 16(F). Last passing=8, First failing=16.
    # Phase 2: Binary search in [8, 15].
    # Mid 12(F). Range [8, 11].
    # Mid 10(P). Best=10. Range [10, 11].
    # Mid 11(F). Range [10, 10].
    # Result: 10.
    
    def is_passing(x: float) -> bool:
        return x <= 10
        
    best, iters = _adaptive_capacity_search(
        start_value=2,
        max_value=20,
        expansion_factor=2.0,
        is_passing=is_passing,
        max_iterations=20,
        precision=0,
        integer_domain=True
    )
    
    assert best == 10
    assert iters < 20


@pytest.mark.unit
def test_adaptive_search_binary_logic_float() -> None:
    """Test the adaptive search logic with float domain (e.g. rate)."""
    
    # Passing: x <= 5.5. Start: 1.0. Max 10.0. Expansion 2.0.
    # Phase 1: 1.0(P) -> 2.0(P) -> 4.0(P) -> 8.0(F). Last=4.0, First=8.0.
    # Phase 2: Binary search [4.0, 8.0] precision 1 (integers 40 to 80).
    # 60 (6.0) -> Fail -> [40, 59]
    # 50 (5.0) -> Pass -> [50, 59], Best=5.0
    # 55 (5.5) -> Pass -> [55, 59], Best=5.5
    # 58 (5.8) -> Fail -> [55, 57]
    # 56 (5.6) -> Fail -> [55, 55]
    # Result -> 5.5
    
    def is_passing(x: float) -> bool:
        return x <= 5.55 # slightly lenient for float comparison, effectively testing 5.5
    
    best, iters = _adaptive_capacity_search(
        start_value=1.0,
        max_value=10.0,
        expansion_factor=2.0,
        is_passing=is_passing,
        max_iterations=20,
        precision=1,
        integer_domain=False
    )
    
    assert best == 5.5


@pytest.mark.unit
def test_patch_traffic_knob_rate() -> None:
    """Test patching rate traffic config."""
    # Setup benchmark config with RateTraffic
    interval = FixedIntervalGeneratorConfig(interval=1.0) # 1.0s -> 1.0 rps
    traffic = RateTrafficConfig(interval_generator=interval)
    config = BenchmarkConfig(traffic_scheduler=traffic)
    
    # Patch to 10.0 rps -> interval 0.1s
    new_config = patch_traffic_knob(config, value=10.0)
    
    new_traffic = new_config.traffic_scheduler
    assert isinstance(new_traffic, RateTrafficConfig)
    assert isinstance(new_traffic.interval_generator, FixedIntervalGeneratorConfig)
    assert new_traffic.interval_generator.interval == pytest.approx(0.1)


@pytest.mark.unit
def test_patch_traffic_knob_concurrent() -> None:
    """Test patching concurrent traffic config."""
    traffic = ConcurrentTrafficConfig(target_concurrent_sessions=1)
    config = BenchmarkConfig(traffic_scheduler=traffic)
    
    # Patch to 5.0
    new_config = patch_traffic_knob(config, value=5.0)
    
    new_traffic = new_config.traffic_scheduler
    assert isinstance(new_traffic, ConcurrentTrafficConfig)
    assert new_traffic.target_concurrent_sessions == 5
    assert new_traffic.rampup_seconds == 5 # Should auto-set rampup
    
    # Patch with non-integer -> ValueError
    with pytest.raises(ValueError):
        patch_traffic_knob(config, value=5.5)

@pytest.mark.unit
@patch("veeksha.capacity_search._read_slo_results")
@patch("veeksha.capacity_search.manage_benchmark_run")
@patch("veeksha.capacity_search._init_capacity_search_output_dir")
def test_run_capacity_search_e2e_flow(
    mock_init_dir,
    mock_manage_run,
    mock_read_slo,
    tmp_path
) -> None:
    """Test the full flow of run_capacity_search logic with mocks."""
    
    # Config
    start_value = 1
    max_value = 5
    
    # Passing criteria: <= 3 is pass, > 3 is fail.
    # 1(P) -> 2(P) -> 4(F). Binary [2, 3]. 3(P). Result 3.
    
    def side_effect_manage_run(cfg):
        # The value passed to patch_traffic_knob is embedded in cfg
        # But run_capacity_search calls patch_traffic_knob internally using the 'value' passed to run_one
        pass

    def side_effect_read_slo(output_dir):
        # We need to know what value was just run.
        # We can look at the latest call to manage_benchmark_run
        # The argument to manage_benchmark_run is the patched config.
        # We can inspect its traffic scheduler.
        
        args, _ = mock_manage_run.call_args
        run_cfg = args[0]
        val = run_cfg.traffic_scheduler.target_concurrent_sessions
        
        # 1(P) -> 2(P) -> 4(F). Binary [2, 3]. Mid 3(P). Result 3.
        if val <= 3:
            return {"all_slos_met": True}
        else:
            return {"all_slos_met": False}
    
    mock_read_slo.side_effect = side_effect_read_slo
    mock_init_dir.return_value = str(tmp_path)
    
    target_traffic = ConcurrentTrafficConfig(target_concurrent_sessions=1)
    base_config = BenchmarkConfig(traffic_scheduler=target_traffic)
    cap_config = CapacitySearchConfig(
        benchmark_config=base_config,
        start_value=start_value,
        max_value=max_value,
        expansion_factor=2.0,
        output_dir=str(tmp_path)
    )
    
    result = run_capacity_search(cap_config)
    
    assert result["best_value"] == 3
    assert result["max_iterations"] == 20
    assert len(result["history"]) == 5
