# Unified DataMiner Strategy Adapter

Raw data scraping and web search were scattered across six scraper scripts in `src/tools/` with duplicated caching, error-handling, and hardcoded fallback chains. We decided to define a unified `SourceProvider` protocol and encapsulate source orchestration inside a `CompositeDataMiner` class. Individual source providers (Reddit, Hacker News, Product Hunt, Tavily, YouTube) implement the same interface and are coordinated behind a single `mine(domain, limit)` method with standardized caching and fallback handling.
