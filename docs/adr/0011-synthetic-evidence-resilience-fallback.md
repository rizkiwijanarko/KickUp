# Synthetic Evidence Resiliency Fallback

When external APIs (Reddit, Hacker News, Tavily) are firewalled, rate-limited, or unavailable, failing immediately degrades user experience during demos, air-gapped evaluation, and local development. We decided to add an automatic Synthetic Evidence Mode fallback in `CompositeDataMiner`: when all live providers return 0 items within the SLA budget, the miner synthesizes structured domain evidence with verified schemas and records an explicit `[SYNTHETIC EVIDENCE MODE]` run event to maintain transparency.
