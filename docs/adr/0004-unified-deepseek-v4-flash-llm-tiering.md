# Unified DeepSeek V4 Flash LLM Tiering

Maintaining disparate LLM providers or fragmented reasoning tiers increases routing complexity and latency variability. We decided to standardize on OpenRouter's frontier model `deepseek/deepseek-v4-flash-0731` across both heavy reasoning tasks (Scorer, Critic) and generative tasks (Pain Point Miner, Idea Generator, Pitch Writer). This provides high-throughput generation, low inference costs, and reliable structured output formatting without requiring multi-provider switching heuristics.
