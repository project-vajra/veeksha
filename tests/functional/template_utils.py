"""Utilities for rendering Jinja templates in tests."""

from pathlib import Path
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader


def render_config_template(template_name: str, **kwargs: Any) -> str:
    """Render a Jinja template with the given context."""
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_name)
    string = template.render(**kwargs)
    print(string)
    return string


def create_benchmark_config(
    model: str,
    output_dir: str,
    api_url: str = "",
    api_key: str = "",
    timeout: int = 60,
    max_completed_requests: int = 5,
    num_clients: int = 1,
    num_concurrent_requests_per_client: int = 1,
    request_generator_type: str = "synthetic",
    length_generator_type: str = "fixed",
    interval_generator_type: str = "static",
    prefill_tokens: int = 50,
    decode_tokens: int = 20,
    duration: float = 1.0,
    qps: float = 1.0,
    cv: float = 0.5,
    min_tokens: int = 10,
    max_tokens: int = 100,
    theta: float = 1.0,
    scramble: bool = True,
    prefill_to_decode_ratio: float = 0.5,
    trace_file: str = "",
    ttft_deadline: float = None,
    tbt_deadline: float = None,
) -> str:
    """Create a benchmark config using the template."""
    return render_config_template(
        "benchmark_config.yml.j2",
        model=model,
        output_dir=output_dir,
        api_url=api_url,
        api_key=api_key,
        timeout=timeout,
        max_completed_requests=max_completed_requests,
        num_clients=num_clients,
        num_concurrent_requests_per_client=num_concurrent_requests_per_client,
        request_generator_type=request_generator_type,
        length_generator_type=length_generator_type,
        interval_generator_type=interval_generator_type,
        prefill_tokens=prefill_tokens,
        decode_tokens=decode_tokens,
        duration=duration,
        qps=qps,
        cv=cv,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        theta=theta,
        scramble=scramble,
        prefill_to_decode_ratio=prefill_to_decode_ratio,
        trace_file=trace_file,
        ttft_deadline=ttft_deadline,
        tbt_deadline=tbt_deadline,
    )


def create_capacity_search_config(
    model: str,
    output_dir: str,
    slos: list,
    api_url: str = "",
    api_key: str = "",
    timeout: int = 60,
    max_completed_requests: int = 3,
    max_iterations: int = 2,
    num_clients: int = 1,
    num_concurrent_requests_per_client: int = 1,
    request_generator_type: str = "synthetic",
    prompt_length: int = 30,
    output_length: int = 15,
    interval_generator_config: dict = None,
) -> str:
    """Create a capacity search config using the template."""
    if interval_generator_config is None:
        interval_generator_config = {"type": "static", "duration": 1.0}

    return render_config_template(
        "capacity_search_config.yml.j2",
        model=model,
        output_dir=output_dir,
        slos=slos,
        api_url=api_url,
        api_key=api_key,
        timeout=timeout,
        max_completed_requests=max_completed_requests,
        max_iterations=max_iterations,
        num_clients=num_clients,
        num_concurrent_requests_per_client=num_concurrent_requests_per_client,
        request_generator_type=request_generator_type,
        prompt_length=prompt_length,
        output_length=output_length,
        interval_generator_config=interval_generator_config,
    )