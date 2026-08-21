# Environment & LLM Configuration

Runtime settings are loaded via `src/config.py` (`pydantic-settings`) and `.env`.

## LLM Provider Configuration

- **Provider Agnostic**: Always obtain model clients via `settings.get_llm_config()` and `src.llm.client.get_llm` / `get_structured_llm`. Never hardcode provider SDKs in agent code.
- **Primary Model**: Defaults to OpenRouter `deepseek/deepseek-v4-flash-0731` via `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`.
- **OpenAI alternatives**: Switch those variables to `https://api.openai.com/v1` with `gpt-4o-mini` or `gpt-4o`.
- **Fast / Fallback Models**: `FAST_LLM_*` can select a separate model per generation tier.
- **Prompts**: Keep prompt templates in `PROMPTS.md`, not inlined in agent source files.

## External Data Ingestion

- **Hacker News**: Default primary source; requires no API key.
- **Product Hunt**: Optional source; configured via `PRODUCT_HUNT_TOKEN`.
- **Tavily Web Search**: Used for web search grounding; requires `TAVILY_API_KEY`.
- **Reddit**: Requires `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET`. If unconfigured, the pain point miner automatically falls back to Hacker News and Tavily.
