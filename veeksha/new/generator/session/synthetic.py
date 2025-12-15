from veeksha.new.config.generator.session import SyntheticSessionGeneratorConfig
from veeksha.new.core.seeding import SeedManager
from veeksha.new.core.session import Session, print_session
from veeksha.new.generator.channel.registry import ChannelGeneratorRegistry
from veeksha.new.generator.session.base import BaseSessionGenerator
from veeksha.new.generator.session_graph.registry import SessionGraphGeneratorRegistry


class SyntheticSessionGenerator(BaseSessionGenerator):
    def __init__(
        self, config: SyntheticSessionGeneratorConfig, seed_manager: SeedManager
    ):
        self.config = config
        self.seed_manager = seed_manager
        self.session_graph_seed_manager = seed_manager.child("session_graph")
        self.channel_rng_factory = seed_manager.numpy_factory("channel")

        # get generators
        self.channels = [
            ChannelGeneratorRegistry.get(
                channel.get_type(), channel, rng=self.channel_rng_factory()
            )
            for channel in self.config.channels
        ]
        self.session_graph_generator = SessionGraphGeneratorRegistry.get(
            self.config.session_graph.get_type(),
            self.config.session_graph,
            seed_manager=self.session_graph_seed_manager,
        )

        self.current_session_id = 0  # incremental global session id
        self.current_request_id = 0  # incremental global request id

    def generate_session(self) -> Session:
        session_graph = self.session_graph_generator.generate_session_graph()
        # todo generate requests
        session = Session(
            id=self.current_session_id,
            session_graph=session_graph,
            requests={},
            cancel_session_on_failure=True,
        )
        self.current_session_id += 1
        print_session(session)
        return session

    def capacity(self) -> int:
        return -1
