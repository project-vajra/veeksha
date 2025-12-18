"""OpenAI Chat Completions client for the new Veeksha framework.

Ported and adapted from veeksha/core/llm_clients/openai_chat_completions_client.py
to work with new Request objects and return RequestResult.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, List, Optional

import httpx

from veeksha.logger import init_logger
from veeksha.new.client.base import BaseLLMClient
from veeksha.new.core.request import Request
from veeksha.new.core.response import ChannelResponse, RequestResult
from veeksha.new.core.tokenizer import TokenizerProvider
from veeksha.new.types import ChannelModality

if TYPE_CHECKING:
    from veeksha.new.config.client import OpenAIChatClientConfig

logger = init_logger(__name__)


class OpenAIChatClient(BaseLLMClient):
    """Async client for OpenAI Chat Completions API using httpx.

    Works with new Request objects that have channels instead of prompt tuples.
    """

    def __init__(
        self,
        config: OpenAIChatClientConfig,
        tokenizer_provider: TokenizerProvider,
    ) -> None:
        """Initialize the OpenAI Chat client.

        Args:
            config: Client configuration with model, timeout, etc.
            tokenizer_provider: Provider for tokenizers per modality.
        """
        super().__init__(config)
        self.tokenizer_provider = tokenizer_provider

        # at this point either self.api_base and self.api_key are set or an error is raised
        if not self.api_base.endswith("/"):  # type: ignore
            self.api_base += "/"  # type: ignore
        self.address = self.api_base + self.config.address_append_value  # type: ignore

        self.text_tokenizer_handle = self.tokenizer_provider.for_modality(
            ChannelModality.TEXT
        )

    def _build_text_content_block(self, text_content: Any) -> tuple[dict, int]:
        """Build a text content block for multimodal messages.

        Args:
            text_content: Text content from request channels (string or structured).

        Returns:
            Tuple of (content_block_dict, token_count).
        """
        if isinstance(text_content, str):
            prompt_text = text_content
            prompt_len = len(self.text_tokenizer_handle.encode(prompt_text))
        else:
            # Support for structured text content if needed later
            prompt_text = getattr(text_content, "input_text", str(text_content))
            prompt_len = getattr(
                text_content,
                "input_length",
                len(self.text_tokenizer_handle.encode(prompt_text)),
            )

        return {"type": "text", "text": prompt_text}, prompt_len

    def _build_image_content_block(self, image_content: Any) -> dict:
        """Build an image content block for multimodal messages.

        Args:
            image_content: Image content - expected to be a URL string or base64 data URI.

        Returns:
            Content block dict for the image.
        """
        if isinstance(image_content, str):
            return {"type": "image_url", "image_url": {"url": image_content}}
        else:
            raise NotImplementedError(
                f"Image content type {type(image_content)} not yet supported"
            )

    def _build_audio_content_block(self, audio_content: Any) -> dict:
        """Build an audio content block for multimodal messages.

        Args:
            audio_content: Audio content - expected to be URL or base64 data.

        Returns:
            Content block dict for the audio.
        """
        if isinstance(audio_content, str):
            return {"type": "audio_url", "audio_url": {"url": audio_content}}
        else:
            raise NotImplementedError(
                f"Audio content type {type(audio_content)} not yet supported"
            )

    def _build_video_content_block(self, video_content: Any) -> dict:
        """Build a video content block for multimodal messages.

        Args:
            video_content: Video content - expected to be URL or base64 data URI.

        Returns:
            Content block dict for the video.
        """
        if isinstance(video_content, str):
            return {"type": "video_url", "video_url": {"url": video_content}}
        else:
            raise NotImplementedError(
                f"Video content type {type(video_content)} not yet supported"
            )

    def _build_message_content(self, request: Request) -> tuple[list, int]:
        """Build multimodal message content from request channels.

        Constructs a list of content blocks in OpenAI multimodal format.

        Args:
            request: Request with channels dict mapping modalities to content.

        Returns:
            Tuple of (content_blocks_list, text_token_count).
        """
        content_blocks: List[dict] = []
        text_token_count = 0

        if ChannelModality.TEXT in request.channels:
            text_block, text_token_count = self._build_text_content_block(
                request.channels[ChannelModality.TEXT]
            )
            content_blocks.append(text_block)

        if ChannelModality.IMAGE in request.channels:
            image_block = self._build_image_content_block(
                request.channels[ChannelModality.IMAGE]
            )
            content_blocks.append(image_block)

        if ChannelModality.AUDIO in request.channels:
            audio_block = self._build_audio_content_block(
                request.channels[ChannelModality.AUDIO]
            )
            content_blocks.append(audio_block)

        if ChannelModality.VIDEO in request.channels:
            video_block = self._build_video_content_block(
                request.channels[ChannelModality.VIDEO]
            )
            content_blocks.append(video_block)

        return content_blocks, text_token_count

    # ---------- response processing

    def _process_text_delta(
        self,
        delta_content: str,
        previous_responses: List[str],
        previous_token_count: int,
        most_recent_token_time: float,
        inter_token_times: List[float],
        generated_text: str,
        tokens_received: int,
        max_tokens_limit: Optional[int],
    ) -> tuple[str, int, int, float, List[float], bool]:
        """Process a text delta from the streaming response.

        Args:
            delta_content: Text content from the delta.
            previous_responses: List of previous response chunks.
            previous_token_count: Token count before this chunk.
            most_recent_token_time: Time of last token received.
            inter_token_times: List of inter-token times.
            generated_text: Accumulated generated text.
            tokens_received: Total tokens received so far.
            max_tokens_limit: Maximum tokens limit (if any).

        Returns:
            Tuple of (generated_text, tokens_received, previous_token_count,
                     most_recent_token_time, inter_token_times, should_break).
        """
        import time

        chunk_time = time.monotonic()

        tokens_this_chunk, previous_token_count = self._get_current_tokens_received(
            previous_responses,
            delta_content,
            previous_token_count,
        )

        allowable = tokens_this_chunk
        if isinstance(max_tokens_limit, int):
            allowable = max(
                0,
                min(tokens_this_chunk, max_tokens_limit - tokens_received),
            )

        if allowable > 0:
            inter_token_times.append(chunk_time - most_recent_token_time)
            if allowable > 1:
                inter_token_times.extend([0] * (allowable - 1))
            tokens_received += allowable
            most_recent_token_time = chunk_time
            generated_text += delta_content

        should_break = (
            isinstance(max_tokens_limit, int) and tokens_received >= max_tokens_limit
        )
        return (
            generated_text,
            tokens_received,
            previous_token_count,
            most_recent_token_time,
            inter_token_times,
            should_break,
        )

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

    def _build_channel_responses(
        self,
        success: bool,
        generated_text: str,
        inter_token_times: List[float],
        prompt_len: int,
        tokens_received: int,
        image_data: Optional[Any],
        audio_data: Optional[Any],
        video_data: Optional[Any],
    ) -> dict:
        """Build channel responses for all modalities.

        Args:
            success: Whether the request was successful.
            generated_text: Generated text output.
            inter_token_times: List of inter-token times.
            prompt_len: Number of prompt tokens.
            tokens_received: Number of output tokens.
            image_data: Image response data (if any).
            audio_data: Audio response data (if any).
            video_data: Video response data (if any).

        Returns:
            Dict mapping ChannelModality to ChannelResponse.
        """
        channels = {}

        if not success:
            return channels

        # Text channel (functional)
        if generated_text:
            channels[ChannelModality.TEXT] = ChannelResponse(
                modality=ChannelModality.TEXT,
                content=generated_text,
                metrics={
                    "inter_token_times": inter_token_times,
                    "num_prompt_tokens": prompt_len,
                    "num_output_tokens": tokens_received,
                },
            )

        if image_data is not None:
            channels[ChannelModality.IMAGE] = ChannelResponse(
                modality=ChannelModality.IMAGE,
                content=image_data,
                metrics={},
            )

        if audio_data is not None:
            channels[ChannelModality.AUDIO] = ChannelResponse(
                modality=ChannelModality.AUDIO,
                content=audio_data,
                metrics={},
            )

        if video_data is not None:
            channels[ChannelModality.VIDEO] = ChannelResponse(
                modality=ChannelModality.VIDEO,
                content=video_data,
                metrics={},
            )

        return channels

    def _get_current_tokens_received(
        self,
        previous_responses: List[str],
        current_response: str,
        previous_token_count: int,
    ) -> tuple[int, int]:
        """Calculate tokens received in this chunk."""
        previous_responses.append(current_response)
        joined = "".join(previous_responses)
        current_total = len(self.text_tokenizer_handle.encode(joined))
        tokens_this_chunk = current_total - previous_token_count
        return tokens_this_chunk, current_total

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

    async def send_request(
        self,
        request: Request,
        session_id: int,
    ) -> RequestResult:
        """Send a request to the OpenAI Chat Completions API."""
        timeout = self.config.request_timeout
        content_blocks, prompt_len = self._build_message_content(request)
        dispatched_at = time.monotonic()

        message = [{"role": "user", "content": content_blocks}]
        body = {
            "model": self.config.model,
            "messages": message,
            "stream": True,
        }

        # TODO check that min tokens is functional
        # min_tokens_target = None
        # if min_tokens_target is not None and self.config.min_tokens_param:
        #     body[self.config.min_tokens_param] = min_tokens_target

        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        # Metrics tracking for text
        inter_token_times: List[float] = []
        error_msg: Optional[str] = None
        error_code: Optional[int] = None
        tokens_received = 0
        generated_text = ""
        previous_responses: List[str] = []
        previous_token_count = 0
        most_recent_token_time = time.monotonic()

        # Multimodal response data (skeletons)
        image_data: Optional[Any] = None
        audio_data: Optional[Any] = None
        video_data: Optional[Any] = None

        # TODO make per-request
        max_tokens_limit = None
        if self.config.additional_sampling_params_dict:
            max_tokens_limit = self.config.additional_sampling_params_dict.get(
                "max_completion_tokens"
            )

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", self.address, json=body, headers=headers
                ) as response:
                    response.raise_for_status()

                    async for data in self._process_stream(response):
                        if "error" in data:
                            err = data.get("error") or {}
                            error_msg = err.get("message", "Unknown error")
                            code_value = err.get("code")
                            error_code = (
                                code_value if isinstance(code_value, int) else 400
                            )
                            break

                        delta = data["choices"][0]["delta"]

                        if delta.get("content"):
                            (
                                generated_text,
                                tokens_received,
                                previous_token_count,
                                most_recent_token_time,
                                inter_token_times,
                                should_break,
                            ) = self._process_text_delta(
                                delta["content"],
                                previous_responses,
                                previous_token_count,
                                most_recent_token_time,
                                inter_token_times,
                                generated_text,
                                tokens_received,
                                max_tokens_limit,
                            )
                            if should_break:
                                break

                        # TODO: image deltas
                        image_data = self._process_image_response(delta, image_data)

                        # TODO: audio deltas
                        audio_data = self._process_audio_response(delta, audio_data)

                        # TODO: video deltas
                        video_data = self._process_video_response(delta, video_data)

        except httpx.HTTPStatusError as e:
            error_code = e.response.status_code if e.response else 500
            error_msg = error_msg or str(e)
            logger.warning(f"HTTP Error: status={error_code} msg={error_msg}")
        except httpx.ConnectError as e:
            error_code = 503
            error_msg = error_msg or str(e)
            logger.warning(f"Connection Error: ({error_code}) {error_msg}")
        except httpx.TimeoutException:
            error_code = 408
            error_msg = error_msg or "Request timed out"
            logger.warning(f"Timeout Error: ({error_code}) {error_msg}")
        except Exception as e:
            error_code = error_code or 520
            error_msg = error_msg or str(e)
            logger.exception(f"Unexpected error: ({error_code}) {error_msg}")

        completed_at = time.monotonic()
        success = error_msg is None and error_code is None

        channels = self._build_channel_responses(
            success=success,
            generated_text=generated_text,
            inter_token_times=inter_token_times,
            prompt_len=prompt_len,
            tokens_received=tokens_received,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
        )

        return RequestResult(
            request_id=request.id,
            session_id=session_id,
            dispatched_at=dispatched_at,
            completed_at=completed_at,
            channels=channels,
            success=success,
            error_code=error_code,
            error_msg=error_msg,
        )
