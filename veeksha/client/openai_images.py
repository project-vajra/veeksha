from __future__ import annotations

import base64
import time
from typing import TYPE_CHECKING, Any, List, Optional

import httpx  # type: ignore

from veeksha.client.openai_base import OpenAIBaseClient
from veeksha.core.request import Request
from veeksha.core.request_content import (
    AudioChannelRequestContent,
    ImageChannelRequestContent,
    TextChannelRequestContent,
    VideoChannelRequestContent,
)
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.core.tokenizer import TokenizerProvider
from veeksha.logger import init_logger
from veeksha.types import ChannelModality

if TYPE_CHECKING:
    from veeksha.config.client import OpenAIImagesClientConfig

logger = init_logger(__name__)


class OpenAIImagesClient(OpenAIBaseClient):
    """OpenAI-compatible Text to Image generation client.
    Endpoint: /v1/images/generations
    """

    def __init__(
        self,
        config: OpenAIImagesClientConfig,
        tokenizer_provider: TokenizerProvider,
    ) -> None:
        """Initialize the OpenAI Images client.

        Args:
            config: Client configuration with model, timeout, etc.
            tokenizer_provider: Provider for tokenizers per modality.
        """
        super().__init__(config=config, tokenizer_provider=tokenizer_provider)
        self.image_endpoint_addr = str(self.api_base) + str(
            self.config.address_append_value
        )
        self.is_stream = self.config.stream  # type: ignore[attr-defined]

    def _build_text_content_block(
        self, text_content: TextChannelRequestContent
    ) -> tuple[dict, int]:
        """Build a text content block for multimodal messages.

        Args:
            text_content: Text content from request channels.

        Returns:
            Tuple of (content_block_dict, token_count).
        """
        prompt_text = text_content.input_text
        prompt_len = len(self.text_tokenizer_handle.encode(prompt_text))

        return {"type": "text", "text": prompt_text}, prompt_len

    def _build_image_content_block(
        self, image_content: ImageChannelRequestContent
    ) -> dict:
        """Build an image content block for multimodal messages."""
        return {"type": "image_url", "image_url": {"url": image_content.input_image}}

    def _build_audio_content_block(
        self, audio_content: AudioChannelRequestContent
    ) -> dict:
        """Build an audio content block for multimodal messages."""
        return {"type": "audio_url", "audio_url": {"url": audio_content.input_audio}}

    def _build_video_content_block(
        self, video_content: VideoChannelRequestContent
    ) -> dict:
        """Build a video content block for multimodal messages."""
        return {"type": "video_url", "video_url": {"url": video_content.input_video}}

    def _build_message_content(self, request: Request) -> tuple[str, int]:
        """Build multimodal message content from request channels.

        Constructs a list of content blocks in OpenAI multimodal format.

        Args:
            request: Request with channels dict mapping modalities to content.

        Returns:
            Tuple of (prompt_string, text_token_count).
        """
        content_blocks: List[dict] = []
        text_token_count = 0
        if ChannelModality.TEXT in request.channels:
            text_block, text_token_count = self._build_text_content_block(
                request.channels[ChannelModality.TEXT]  # type: ignore
            )
            content_blocks.append(text_block)

        if ChannelModality.IMAGE in request.channels:
            image_block = self._build_image_content_block(
                request.channels[ChannelModality.IMAGE]  # type: ignore
            )
            content_blocks.append(image_block)

        if ChannelModality.AUDIO in request.channels:
            audio_block = self._build_audio_content_block(
                request.channels[ChannelModality.AUDIO]  # type: ignore
            )
            content_blocks.append(audio_block)

        if ChannelModality.VIDEO in request.channels:
            video_block = self._build_video_content_block(
                request.channels[ChannelModality.VIDEO]  # type: ignore
            )
            content_blocks.append(video_block)

        messages = []
        if request.history:
            messages.extend(request.history)

        # current request content
        if len(content_blocks) == 1 and content_blocks[0].get("type") == "text":
            messages.append({"role": "user", "content": content_blocks[0]["text"]})
        else:
            messages.append({"role": "user", "content": content_blocks})

        # Extract all text content from entire history + current request for the prompt
        prompt = ""
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                # Simple text content
                prompt += content + " "
            elif isinstance(content, list):
                # Multimodal content - extract text blocks
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        prompt += block.get("text", "") + " "

        prompt = prompt.strip()
        return prompt, text_token_count

    def _process_image_response(
        self,
        delta: dict,
        image_data: Optional[Any],
    ) -> Optional[Any]:
        """Process image data from the streaming response.

        Args:
            delta: Delta dict from the streaming response.
            image_data: Accumulated image data (if any).

        Returns:
            Updated image data or None.
        """
        # Skeleton: Log warning and return None
        # Future: Parse image data from delta and accumulate
        return None

    def _process_audio_response(
        self,
        delta: dict,
        audio_data: Optional[Any],
    ) -> Optional[Any]:
        """Process audio data from the streaming response.

        Args:
            delta: Delta dict from the streaming response.
            audio_data: Accumulated audio data (if any).

        Returns:
            Updated audio data or None.
        """
        # Skeleton: Log warning and return None
        # Future: Parse audio chunks from delta and accumulate
        return None

    def _process_video_response(
        self,
        delta: dict,
        video_data: Optional[Any],
    ) -> Optional[Any]:
        """Process video data from the streaming response.

        Args:
            delta: Delta dict from the streaming response.
            video_data: Accumulated video data (if any).

        Returns:
            Updated video data or None.
        """
        # Skeleton: Log warning and return None
        # Future: Parse video data from delta and accumulate
        return None

    async def _process_stream(self, response: httpx.Response):
        """Process SSE stream from server."""
        import json

        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        return
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

    def __build_image_response(
        self,
        success: bool,
        imgs: List[Any],
        inter_chunk_times: List[float],
        num_total_prompt_tokens: int,
        delta_prompt_len: int,
    ) -> ChannelResponse:
        """Build the ChannelResponse for image data.

        Args:
            success: Whether the request was successful.
            imgs: List of generated images.
            inter_chunk_times: List of times between chunks.
            num_total_prompt_tokens: Total number of prompt tokens.
            delta_prompt_len: Change in prompt length.
        """
        metrics = {
            "is_stream": self.is_stream,
            "inter_chunk_times": inter_chunk_times,
            "num_total_prompt_tokens": num_total_prompt_tokens,
            "num_output_images": len(imgs),
            "num_delta_prompt_tokens": delta_prompt_len,
        }
        return ChannelResponse(
            modality=ChannelModality.IMAGE,
            content=imgs,
            metrics=metrics,
        )

    async def send_request(
        self,
        request: Request,
        session_id: int,
        session_total_requests: int = 1,
    ) -> RequestResult:
        """Send a request to the OpenAI Images endpoint."""

        timeout = self.config.request_timeout
        num_images = self.config.num_images  # type: ignore[attr-defined]
        size = self.config.size  # type: ignore[attr-defined]
        quality = "auto"
        if (
            request.requested_output is not None
            and request.requested_output.image is not None
        ):
            num_images = (
                request.requested_output.image.num_images
                if request.requested_output.image.num_images is not None
                else num_images
            )
            size = (
                request.requested_output.image.size
                if request.requested_output.image.size is not None
                else size
            )
            quality = (
                request.requested_output.image.quality
                if request.requested_output.image.quality is not None
                else quality
            )
        # image metrics

        error_msg: Optional[str] = None
        error_code: Optional[int] = None
        chunks_received = 0

        # multimodal response data
        image_data: Optional[Any] = None
        audio_data: Optional[Any] = None
        video_data: Optional[Any] = None
        imgs: List[Any] = []

        prompt = ""
        start_time = time.monotonic()
        completed_at = time.monotonic()
        delta_prompt_len = 0
        try:
            prompt, delta_prompt_len = self._build_message_content(request)
            body = {
                "prompt": prompt,
                "model": self.config.model,
                "n": num_images,
                "size": size,
                "response_format": self.config.response_format,  # type: ignore[attr-defined]
                "quality": quality,
            }
            body.update(self._get_sampling_params(request))

            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            client = self._get_client()
            if not self.is_stream:
                # Non-streaming request
                response = await client.post(
                    self.image_endpoint_addr,
                    json=body,
                    headers=headers,
                    timeout=timeout,
                )
                response.raise_for_status()
                data = response.json()
                completed_at = time.monotonic()
                images_generated = data.get("data", [])
                for img in images_generated:
                    if img.get("b64_json"):
                        b64_data = img["b64_json"]
                        decoded_image = base64.b64decode(b64_data)
                        imgs.append(decoded_image)
                    elif img.get("url"):
                        # Handle URL format if needed
                        imgs.append(img["url"])
            else:
                raise NotImplementedError(
                    "Streaming not implemented for image generation yet."
                )
        except httpx.HTTPStatusError as e:
            error_code = e.response.status_code if e.response else 500
            error_msg = error_msg or str(e)
            logger.warning("HTTP Error: status=%s msg=%s", error_code, error_msg)
        except httpx.ConnectError as e:
            error_code = 503
            error_msg = error_msg or str(e)
            logger.warning("Connection Error: (%s) %s", error_code, error_msg)
        except httpx.TimeoutException:
            error_code = 408
            error_msg = error_msg or "Request timed out"
            logger.warning("Timeout Error: (%s) %s", error_code, error_msg)
        except Exception as e:
            error_code = error_code or 520
            error_msg = error_msg or str(e)
            logger.warning("Unexpected Error: (%s) %s", error_code, error_msg)

        success = error_msg is None and error_code is None
        inter_chunk_times: List[float] = []
        num_total_prompt_tokens = 0
        if success:
            end_to_end = max(0.0, completed_at - start_time)
            inter_chunk_times = [end_to_end]
            num_total_prompt_tokens = self._get_cached_token_count(prompt)
        channel_responses = {}
        channel_responses[ChannelModality.IMAGE] = self.__build_image_response(
            success=success,
            imgs=imgs,
            inter_chunk_times=inter_chunk_times,
            num_total_prompt_tokens=num_total_prompt_tokens,
            delta_prompt_len=delta_prompt_len,
        )

        return RequestResult(
            request_id=request.id,
            session_id=session_id,
            session_total_requests=session_total_requests,
            channels=channel_responses,
            success=success,
            error_code=error_code,
            error_msg=error_msg,
            client_completed_at=completed_at,
        )
