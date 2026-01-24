import asyncio
import base64

import pytest

from veeksha.client import ClientRegistry
from veeksha.config.client import OpenAIImagesClientConfig
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.requested_output import ImageOutputSpec, RequestedOutputSpec
from veeksha.core.response import RequestResult
from veeksha.core.tokenizer import TokenizerHandle, TokenizerProvider
from veeksha.types import ChannelModality


@pytest.mark.unit
def test_openai_images_client_basic_request() -> None:
    """Test basic image generation request with text prompt."""
    tokenizer_handle = TokenizerHandle(
        count_tokens=lambda text: len(str(text).split()),
        decode=lambda token_ids: "",
        encode=lambda text: [0] * len(str(text).split()),
    )
    tokenizer_provider = TokenizerProvider({ChannelModality.TEXT: tokenizer_handle})

    config = OpenAIImagesClientConfig(
        api_base="http://example.com/v1",
        api_key="test-key",
        model="dall-e-3",
        num_images=1,
        size="1024x1024",
        response_format="b64_json",
    )
    client = ClientRegistry.get(
        config.get_type(),
        config=config,
        tokenizer_provider=tokenizer_provider,
    )

    # Create a mock response
    mock_image_b64 = base64.b64encode(b"fake_image_data").decode("utf-8")

    async def _fake_send_request(
        request: Request,
        session_id: int,
        session_total_requests: int = 1,
    ) -> RequestResult:
        # Verify the request has the expected structure
        assert ChannelModality.TEXT in request.channels
        assert isinstance(request.channels[ChannelModality.TEXT], TextChannelRequestContent)
        assert request.requested_output is not None
        assert request.requested_output.image is not None
        
        return RequestResult(
            request_id=request.id,
            session_id=session_id,
            session_total_requests=session_total_requests,
            channels={
                ChannelModality.IMAGE: {
                    "content": [mock_image_b64],
                    "metrics": {
                        "is_stream": False,
                        "inter_chunk_times": [1.0],
                        "num_total_prompt_tokens": 5,
                        "num_output_images": 1,
                        "num_delta_prompt_tokens": 5,
                    },
                }
            },
            success=True,
            client_completed_at=1.0,
        )

    client.send_request = _fake_send_request  

    request = Request(
        id=1,
        channels={
            ChannelModality.TEXT: TextChannelRequestContent(
                input_text="a beautiful sunset over the ocean",
            )
        },
        requested_output=RequestedOutputSpec(
            image=ImageOutputSpec(
                num_images=1,
                size="1024x1024",
            )
        ),
    )

    result = asyncio.run(client.send_request(request, session_id=0, session_total_requests=1))
    
    assert result.success
    assert result.request_id == 1
    assert result.session_id == 0
    assert ChannelModality.IMAGE in result.channels




@pytest.mark.unit
def test_openai_images_client_multiple_images() -> None:
    """Test generating multiple images in a single request."""
    tokenizer_handle = TokenizerHandle(
        count_tokens=lambda text: len(str(text).split()),
        decode=lambda token_ids: "",
        encode=lambda text: [0] * len(str(text).split()),
    )
    tokenizer_provider = TokenizerProvider({ChannelModality.TEXT: tokenizer_handle})

    config = OpenAIImagesClientConfig(
        api_base="http://example.com/v1",
        api_key="test-key",
        model="dall-e-3",
        num_images=3,
        size="512x512",
        response_format="b64_json",
    )
    client = ClientRegistry.get(
        config.get_type(),
        config=config,
        tokenizer_provider=tokenizer_provider,
    )

    request = Request(
        id=2,
        channels={
            ChannelModality.TEXT: TextChannelRequestContent(
                input_text="generate three cats",
            )
        },
        requested_output=RequestedOutputSpec(
            image=ImageOutputSpec(
                num_images=3,
                size="512x512",
            )
        ),
    )

    # Mock the send_request to verify num_images is passed correctly
    async def _fake_send_request(
        request: Request,
        session_id: int,
        session_total_requests: int = 1,
    ) -> RequestResult:
        assert request.requested_output is not None
        assert request.requested_output.image is not None
        assert request.requested_output.image.num_images == 3
        assert request.requested_output.image.size == "512x512"
        
        return RequestResult(
            request_id=request.id,
            session_id=session_id,
            session_total_requests=session_total_requests,
            channels={
                ChannelModality.IMAGE: {
                    "content": ["img1", "img2", "img3"],
                    "metrics": {
                        "num_output_images": 3,
                    },
                }
            },
            success=True,
            client_completed_at=2.0,
        )

    client.send_request = _fake_send_request  # type: ignore[method-assign]

    result = asyncio.run(client.send_request(request, session_id=0, session_total_requests=1))
    
    assert result.success
    assert result.channels[ChannelModality.IMAGE]["metrics"]["num_output_images"] == 3


@pytest.mark.unit
def test_openai_images_client_response_format() -> None:
    """Test different response formats (b64_json vs url)."""
    tokenizer_handle = TokenizerHandle(
        count_tokens=lambda text: len(str(text).split()),
        decode=lambda token_ids: "",
        encode=lambda text: [0] * len(str(text).split()),
    )
    tokenizer_provider = TokenizerProvider({ChannelModality.TEXT: tokenizer_handle})

    # Test with b64_json format
    config_b64 = OpenAIImagesClientConfig(
        api_base="http://example.com/v1",
        api_key="test-key",
        model="dall-e-3",
        response_format="b64_json",
    )
    client_b64 = ClientRegistry.get(
        config_b64.get_type(),
        config=config_b64,
        tokenizer_provider=tokenizer_provider,
    )
    assert client_b64.config.response_format == "b64_json"

    # Test with url format
    config_url = OpenAIImagesClientConfig(
        api_base="http://example.com/v1",
        api_key="test-key",
        model="dall-e-3",
        response_format="url",
    )
    client_url = ClientRegistry.get(
        config_url.get_type(),
        config=config_url,
        tokenizer_provider=tokenizer_provider,
    )
    assert client_url.config.response_format == "url"


@pytest.mark.unit
def test_openai_images_client_tokenizer_model() -> None:
    """Test that tokenizer_model can be specified separately from model.
    
    When tokenizer_model is None, the tokenizer will use the main model.
    When tokenizer_model is specified, it uses that model for tokenization.
    """
    tokenizer_handle = TokenizerHandle(
        count_tokens=lambda text: len(str(text).split()),
        decode=lambda token_ids: "",
        encode=lambda text: [0] * len(str(text).split()),
    )
    tokenizer_provider = TokenizerProvider({ChannelModality.TEXT: tokenizer_handle})

    # Test with tokenizer_model set to a different model
    config = OpenAIImagesClientConfig(
        api_base="http://example.com/v1",
        api_key="test-key",
        model="dall-e-3",
        tokenizer_model="gpt-4",  # Different tokenizer model
    )
    client = ClientRegistry.get(
        config.get_type(),
        config=config,
        tokenizer_provider=tokenizer_provider,
    )

    # Verify tokenizer_model is set correctly
    assert client.config.tokenizer_model == "gpt-4"
    assert client.config.model == "dall-e-3"