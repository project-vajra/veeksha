from abc import abstractmethod

from veeksha.new.config.generator.session import BaseSessionGeneratorConfig
from veeksha.new.core.session import Session


class BaseSessionGenerator:
    def __init__(self, config: BaseSessionGeneratorConfig):
        self.config = config

    @abstractmethod
    def generate_session(self) -> Session:
        pass

    @abstractmethod
    def capacity(self) -> int:
        """Total number of sessions producible if finite; -1 if unbounded."""
