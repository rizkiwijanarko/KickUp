# Declarative LangGraph Reflection Loop

The previous architecture routed all execution, revisions, and loops through a single 800-line procedural orchestrator node that manually tracked indices, checked stage flags, and mutated state fields. We decided to replace the imperative orchestrator node with native LangGraph conditional edges originating directly from the Critic node to target worker nodes (`pain_point_miner`, `idea_generator`, `pitch_writer`). This eliminates the orchestrator bottleneck and delegates lifecycle transitions directly to the compiled StateGraph.
