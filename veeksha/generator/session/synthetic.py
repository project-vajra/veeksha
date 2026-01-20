from veeksha.config.generator.session import SyntheticSessionGeneratorConfig
from veeksha.core.request import Request
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.core.session_graph import get_node_ids, is_root, parents
from veeksha.core.tokenizer import TokenizerProvider
from veeksha.generator.channel.registry import ChannelGeneratorRegistry
from veeksha.generator.output_spec import OutputSpecGenerator
from veeksha.generator.session.base import BaseSessionGenerator
from veeksha.generator.session_graph.registry import SessionGraphGeneratorRegistry
from veeksha.types import ChannelModality


class SyntheticSessionGenerator(BaseSessionGenerator):
    """Generates synthetic sessions with configurable input content and output specs.

    This generator creates sessions with:
    - Session graphs defining the request structure (linear, branching, etc.)
    - Input content for each channel (text, image, etc.)
    - Output specifications defining expected model output
    """

    def __init__(
        self,
        config: SyntheticSessionGeneratorConfig,
        seed_manager: SeedManager,
        tokenizer_provider: TokenizerProvider,
        append_min_tokens_instruction: bool = False,
    ):
        self.config = config
        self.seed_manager = seed_manager
        self.tokenizer_provider = tokenizer_provider
        self.append_min_tokens_instruction = append_min_tokens_instruction

        # channel generators
        self.channels = {}
        for channel in self.config.channels:
            tokenizer_handle = self.tokenizer_provider.for_modality(channel.get_type())
            channel_kwargs = {
                "seed_manager": self.seed_manager.child(
                    f"channel_{channel.get_type()}"
                ),
                "tokenizer_handle": tokenizer_handle,
            }

            self.channels[channel.get_type()] = ChannelGeneratorRegistry.get(
                channel.get_type(),
                channel,
                **channel_kwargs,
            )

        self.session_graph_generator = SessionGraphGeneratorRegistry.get(
            self.config.session_graph.get_type(),
            self.config.session_graph,
            seed_manager=seed_manager.child("session_graph"),
        )

        self.output_spec_generator = OutputSpecGenerator(
            self.config.output_spec,
            seed_manager.child("output_spec"),
        )

        self.current_session_id = 0  # incremental global session id
        self.current_request_id = 0  # incremental global request id

    def generate_session(self) -> Session:
        session_graph = self.session_graph_generator.generate_session_graph()
        requests = {}

        for node_id in get_node_ids(session_graph):
            requested_output = self.output_spec_generator.generate()

            # min_tokens_suffix for text channel if needed
            min_tokens_suffix = None
            if (
                self.append_min_tokens_instruction
                and requested_output is not None
                and requested_output.text is not None
            ):
                min_tokens_suffix = requested_output.text.target_tokens

            channels = {}
            for channel_type, channel in self.channels.items():
                if channel_type == ChannelModality.TEXT:
                    channels[channel_type] = channel.generate_content(
                        is_root=is_root(session_graph, node_id),
                        min_tokens_suffix=min_tokens_suffix,
                    )
                else:
                    channels[channel_type] = channel.generate_content(
                        is_root=is_root(session_graph, node_id)
                    )

            incoming_edges = parents(session_graph, node_id)
            history_parents = [e for e in incoming_edges if e.is_history_parent]
            session_context = {
                "node_id": node_id,
                "wait_after_ready": session_graph.nodes[node_id].wait_after_ready,
                "parent_nodes": [e.src for e in incoming_edges],
                "history_parent": history_parents[0].src if history_parents else None,
            }

            request = Request(
                id=self.current_request_id,
                channels=channels,
                session_context=session_context,
                requested_output=requested_output,
            )
            requests[node_id] = request
            self.current_request_id += 1

        session = Session(
            id=self.current_session_id,
            session_graph=session_graph,
            requests=requests,
        )
        self.current_session_id += 1
        return session

    def capacity(self) -> int:
        return -1
