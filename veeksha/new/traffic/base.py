from veeksha.new.config.traffic import BaseTrafficConfig
from veeksha.new.core.seeding import SeedManager


class BaseTrafficScheduler:
    def __init__(self, config: BaseTrafficConfig, seed_manager: SeedManager):
        self.config = config
        self.seed_manager = seed_manager
