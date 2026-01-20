from typing import Optional

from veeksha.config.generator.requested_output import OutputSpecConfig
from veeksha.core.requested_output import (
    AudioOutputSpec,
    ImageOutputSpec,
    RequestedOutputSpec,
    TextOutputSpec,
    VideoOutputSpec,
)
from veeksha.core.seeding import SeedManager
from veeksha.generator.length.registry import LengthGeneratorRegistry


class OutputSpecGenerator:
    """Generates RequestedOutputSpec instances from configuration.

    This generator handles the dynamic generation of output specifications
    for each request. i.e. for text output, it uses a length generator to produce
    varying output token targets.

    Example:
        >>> config = OutputSpecConfig(
        ...     text=TextOutputSpecConfig(
        ...         output_length_generator=UniformLengthGeneratorConfig(min=100, max=500)
        ...     )
        ... )
        >>> generator = OutputSpecGenerator(config, seed_manager)
        >>> spec = generator.generate()
        >>> spec.text.target_tokens  # between 100-500
    """

    def __init__(
        self,
        config: OutputSpecConfig,
        seed_manager: SeedManager,
    ):
        """Initialize the output spec generator.

        Args:
            config: Output specification configuration.
            seed_manager: Seed manager for reproducible random generation.
        """
        self.config = config
        self.seed_manager = seed_manager

        self._text_length_generator = None
        if config.text is not None:
            self._text_length_generator = LengthGeneratorRegistry.get(
                config.text.output_length_generator.get_type(),
                config.text.output_length_generator,
                rng=seed_manager.numpy_factory("output_length")(),
            )

    def generate(self) -> RequestedOutputSpec:
        """Generate a RequestedOutputSpec for a single request.

        Returns:
            RequestedOutputSpec with configured output specifications.
        """
        text_spec: Optional[TextOutputSpec] = None
        image_spec: Optional[ImageOutputSpec] = None
        audio_spec: Optional[AudioOutputSpec] = None
        video_spec: Optional[VideoOutputSpec] = None

        # text output spec
        if self.config.text is not None and self._text_length_generator is not None:
            target_tokens = self._text_length_generator.get_next_value()
            text_spec = TextOutputSpec(target_tokens=target_tokens)

        # image output spec (static config, no dynamic generation)
        if self.config.image is not None:
            image_spec = ImageOutputSpec(
                num_images=self.config.image.num_images,
                size=self.config.image.size,
                quality=self.config.image.quality,
            )

        # audio output spec (static config, no dynamic generation)
        if self.config.audio is not None:
            raise NotImplementedError
            # audio_spec = AudioOutputSpec()

        # video output spec (static config, no dynamic generation)
        if self.config.video is not None:
            raise NotImplementedError
            # video_spec = VideoOutputSpec(
            #     duration_seconds=self.config.video.duration_seconds,
            #     fps=self.config.video.fps,
            #     resolution=self.config.video.resolution,
            # )

        return RequestedOutputSpec(
            text=text_spec,
            image=image_spec,
            audio=audio_spec,
            video=video_spec,
        )
