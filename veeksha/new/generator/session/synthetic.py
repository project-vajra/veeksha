from veeksha.new.config.generator.session import SyntheticSessionGeneratorConfig
from veeksha.new.core.session import Session
from veeksha.new.generator.channel.registry import ChannelGeneratorRegistry
from veeksha.new.generator.session.base import BaseSessionGenerator
from veeksha.new.generator.session_graph.registry import SessionGraphGeneratorRegistry


class SyntheticSessionGenerator(BaseSessionGenerator):
    def __init__(self, config: SyntheticSessionGeneratorConfig):
        self.config = config
        self.channels = [
            ChannelGeneratorRegistry.get(channel.get_type(), channel)
            for channel in self.config.channels
        ]
        self.session_graph_generator = SessionGraphGeneratorRegistry.get(
            self.config.session_graph.get_type(), self.config.session_graph
        )

    def generate_session(self) -> Session:
        # TODO
        return Session(
            session_id=0,
            session_total_requests=1,
            cancel_session_on_failure=True,
            requests=[],
        )

    def capacity(self) -> int:
        return -1
