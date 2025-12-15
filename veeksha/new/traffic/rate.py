from veeksha.new.config.traffic import RateTrafficConfig
from veeksha.new.core.seeding import SeedManager
from veeksha.new.traffic.base import BaseTrafficScheduler


class RateTrafficScheduler(BaseTrafficScheduler):
    def __init__(self, config: RateTrafficConfig, seed_manager: SeedManager):
        self.config = config
        self.seed_manager = seed_manager
