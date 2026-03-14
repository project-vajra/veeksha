from veeksha.benchmark_utils import maybe_run_server_warmup
from veeksha.config.runtime import RuntimeConfig, WarmupRequestConfig
from veeksha.core.response import RequestResult
from veeksha.types import ChannelModality


class _FakeTokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()


class _FakeClient:
    def __init__(self, *, success: bool = True) -> None:
        self.text_tokenizer_handle = _FakeTokenizer()
        self.success = success
        self.calls = []

    async def send_request(self, request, session_id: int, session_total_requests=1):
        self.calls.append((request, session_id, session_total_requests))
        return RequestResult(
            request_id=request.id,
            session_id=session_id,
            session_total_requests=session_total_requests,
            channels={ChannelModality.TEXT: None},  # type: ignore[dict-item]
            success=self.success,
            error_code=None if self.success else 500,
            error_msg=None if self.success else "warmup failed",
        )


def test_maybe_run_server_warmup_sends_one_request_when_enabled() -> None:
    client = _FakeClient()
    runtime = RuntimeConfig(
        warmup_request=WarmupRequestConfig(
            enabled=True,
            prompt="hello warmup",
            output_tokens=7,
        )
    )

    maybe_run_server_warmup(runtime, client)

    assert len(client.calls) == 1
    request, session_id, session_total_requests = client.calls[0]
    assert session_id == -1
    assert session_total_requests == 1
    assert request.channels[ChannelModality.TEXT].input_text == "hello warmup"
    assert request.channels[ChannelModality.TEXT].target_prompt_tokens == 2
    assert request.requested_output.text.target_tokens == 7


def test_maybe_run_server_warmup_skips_when_disabled() -> None:
    client = _FakeClient()

    maybe_run_server_warmup(RuntimeConfig(), client)

    assert client.calls == []


def test_maybe_run_server_warmup_raises_on_request_failure() -> None:
    client = _FakeClient(success=False)
    runtime = RuntimeConfig(
        warmup_request=WarmupRequestConfig(
            enabled=True,
            prompt="hello warmup",
            output_tokens=7,
        )
    )

    try:
        maybe_run_server_warmup(runtime, client)
    except RuntimeError as exc:
        assert "Warmup request failed" in str(exc)
        assert "500" in str(exc)
        assert "warmup failed" in str(exc)
    else:
        raise AssertionError("Expected warmup failure to raise RuntimeError")
