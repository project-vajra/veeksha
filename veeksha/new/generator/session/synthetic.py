from veeksha.new.config.generator.session import SyntheticSessionGeneratorConfig
from veeksha.new.core.request import Request
from veeksha.new.core.seeding import SeedManager
from veeksha.new.core.session import Session
from veeksha.new.core.session_graph import get_node_ids
from veeksha.new.core.tokenizer import TokenizerProvider
from veeksha.new.generator.channel.registry import ChannelGeneratorRegistry
from veeksha.new.generator.session.base import BaseSessionGenerator
from veeksha.new.generator.session_graph.registry import SessionGraphGeneratorRegistry


class SyntheticSessionGenerator(BaseSessionGenerator):
    def __init__(
        self,
        config: SyntheticSessionGeneratorConfig,
        seed_manager: SeedManager,
        tokenizer_provider: TokenizerProvider,
    ):
        self.config = config
        self.seed_manager = seed_manager
        self.tokenizer_provider = tokenizer_provider

        # get generators
        self.channels = {}
        for channel in self.config.channels:
            tokenizer_handle = self.tokenizer_provider.for_modality(channel.get_type())
            self.channels[channel.get_type()] = ChannelGeneratorRegistry.get(
                channel.get_type(),
                channel,
                seed_manager=self.seed_manager.child(f"channel_{channel.get_type()}"),
                tokenizer_handle=tokenizer_handle,
            )
        self.session_graph_generator = SessionGraphGeneratorRegistry.get(
            self.config.session_graph.get_type(),
            self.config.session_graph,
            seed_manager=seed_manager.child("session_graph"),
        )

        self.current_session_id = 0  # incremental global session id
        self.current_request_id = 0  # incremental global request id

    def generate_session(self) -> Session:
        session_graph = self.session_graph_generator.generate_session_graph()
        requests = {}

        for node_id in get_node_ids(session_graph):
            channels = {}
            for channel_type, channel in self.channels.items():
                channels[channel_type] = channel.generate_content()
            request = Request(
                id=self.current_request_id,
                channels=channels,
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
