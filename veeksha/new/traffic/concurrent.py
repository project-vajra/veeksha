from veeksha.new.config.traffic import ConcurrentTrafficConfig
from veeksha.new.core.seeding import SeedManager
from veeksha.new.traffic.base import BaseTrafficScheduler


class ConcurrentTrafficScheduler(BaseTrafficScheduler):
    def __init__(self, config: ConcurrentTrafficConfig, seed_manager: SeedManager):
        self.config = config
        self.seed_manager = seed_manager
