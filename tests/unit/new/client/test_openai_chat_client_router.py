import asyncio

import pytest

from veeksha.new.client import ClientRegistry
from veeksha.new.config.client import OpenAIRouterClientConfig
from veeksha.new.core.request import Request
from veeksha.new.core.request_content import TextChannelRequestContent
from veeksha.new.core.response import RequestResult
from veeksha.new.core.tokenizer import TokenizerHandle, TokenizerProvider
from veeksha.new.types import ChannelModality


@pytest.mark.unit
def test_openai_router_client_routes_per_request_api_mode() -> None:
    tokenizer_handle = TokenizerHandle(
        count_tokens=lambda text: len(str(text).split()),
        decode=lambda token_ids: "",
        encode=lambda text: [0] * len(str(text).split()),
    )
    tokenizer_provider = TokenizerProvider({ChannelModality.TEXT: tokenizer_handle})

    config = OpenAIRouterClientConfig(
        api_base="http://example.com/v1",
        api_key="",
        model="dummy",
    )
    client = ClientRegistry.get(
        config.get_type(),
        config=config,
        tokenizer_provider=tokenizer_provider,
    )

    called = {"chat": 0, "completions": 0}

    async def _fake_chat_send_request(
        *,
        request: Request,
        session_id: int,
        session_total_requests: int = 1,
    ) -> RequestResult:
        called["chat"] += 1
        return RequestResult(
            request_id=request.id,
            session_id=session_id,
            session_total_requests=session_total_requests,
            success=True,
            client_completed_at=0.0,
        )

    async def _fake_completions_send_request(
        *,
        request: Request,
        session_id: int,
        session_total_requests: int = 1,
    ) -> RequestResult:
        called["completions"] += 1
        return RequestResult(
            request_id=request.id,
            session_id=session_id,
            session_total_requests=session_total_requests,
            success=True,
            client_completed_at=0.0,
        )

    client._chat_client.send_request = _fake_chat_send_request  # type: ignore[attr-defined]
    client._completions_client.send_request = (  # type: ignore[attr-defined]
        _fake_completions_send_request
    )

    chat_req = Request(
        id=1,
        channels={
            ChannelModality.TEXT: TextChannelRequestContent(
                input_text="hello",
                target_output_tokens=1,
            )
        },
        metadata={"api_mode": "chat"},
    )
    completions_req = Request(
        id=2,
        channels={
            ChannelModality.TEXT: TextChannelRequestContent(
                input_text="hello",
                target_output_tokens=1,
            )
        },
        metadata={"api_mode": "completions"},
    )

    asyncio.run(client.send_request(chat_req, session_id=0, session_total_requests=1))
    asyncio.run(
        client.send_request(completions_req, session_id=0, session_total_requests=1)
    )

    assert called["chat"] == 1
    assert called["completions"] == 1


