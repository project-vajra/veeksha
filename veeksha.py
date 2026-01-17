from veeksha.config.generator.session_graph import LinearSessionGraphGeneratorConfig
from veeksha.config.generator.length import FixedLengthGeneratorConfig
from veeksha.generator.session_graph.linear import LinearSessionGraphGenerator
from veeksha.core.seeding import SeedManager

config = LinearSessionGraphGeneratorConfig(
    num_request_generator=FixedLengthGeneratorConfig(value=5),
    inherit_history=True,
)
generator = LinearSessionGraphGenerator(config, SeedManager(42))

graph = generator.generate_session_graph()
print(f"Generated graph with {len(graph.nodes)} nodes")