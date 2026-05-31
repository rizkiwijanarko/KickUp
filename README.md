# 🚀 VentureForge

> An autonomous multi-agent system that discovers startup ideas by mining real user pain points from online communities, evaluating them with Paul Graham-inspired criteria, and generating investor-ready pitch briefs.

**Built for AMD AI Hackathon** | **Track 1: AI Agents & Agentic Workflows**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-green.svg)](https://github.com/langchain-ai/langgraph)
[![AMD ROCm](https://img.shields.io/badge/AMD-ROCm-red.svg)](https://www.amd.com/en/products/software/rocm.html)

**🚀 Try the live demo:** [https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/VentureForge](https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/VentureForge)

---

## 🎯 What is VentureForge?

VentureForge is a **hierarchical multi-agent system** that automates startup idea discovery and validation:

1. **Mines pain points** from Hacker News, Product Hunt, and YouTube Data API
2. **Clusters similar complaints** using LLM-based grouping
3. **Generates startup ideas** that address multiple pain points
4. **Evaluates ideas** using binary yes/no rubrics (Paul Graham framework)
5. **Writes pitch briefs** with competitive intelligence and validation plans
6. **Critiques outputs** with adversarial review and reflection loop (max 2 revisions, configurable)

**Key Innovation:** All evaluation uses **binary yes/no checks** instead of subjective 0-1 scores, making results reproducible and auditable.

---

## 🏛️ Architecture

### Hierarchical + Reflection + Pressure-Test Pattern

```
                    ┌──────────────────┐
                    │   ORCHESTRATOR   │  ← Supervisor (routing + state mgmt)
                    └────────┬─────────┘
                             │
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
        ┌────────┐      ┌────────┐      ┌────────┐
        │ Pain   │      │ Idea   │      │ Scorer │
        │ Point  │ ───► │ Gen    │ ───► │ (8 yes/│
        │ Miner  │      │        │      │ no chks│
        │(4 src) │      │(1-at-  │      │        │
        └────────┘      │ time)  │      └────────┘
                        └────────┘           │
                             │                │
                             ▼                │
                        ┌────────┐           │
                        │ Pitch  │ ───────────┘
                        │ Writer │
                        │(1-at-  │
                        │ time)  │
                        └────────┘
                             │
                             ▼
                        ┌──────────────────┐
                        │   CRITIC AGENT   │
                        │  (5 yes/no chks) │
                        └────────┬─────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
               approve                    revise
                    │                         │
                    ▼                         ▼
                  END                ┌───────┴───────┐
                                    │               │
                              pain_point    idea_generator
                                  miner              │
                                    │               │
                                    └───────┬───────┘
                                            │
                                      pitch_writer
```

### 6 Agents

| Agent | Role | Reflection Target? |
|-------|------|--------------------|
| **Orchestrator** | Central supervisor, routes tasks, manages state, handles reflection loop | — |
| **Pain Point Miner** | Scrapes HN, Product Hunt, YouTube in parallel; clusters complaints | ✅ |
| **Idea Generator** | Groups pain points into themes, generates ideas (one-at-a-time) | ✅ |
| **Scorer** | Evaluates ideas via 8 binary checks + fatal flaw detection | — |
| **Pitch Writer** | Writes one-page briefs (one-at-a-time, compressed prompts) | ✅ |
| **Critic** | Adversarial review with 5 core checks, routes back to target agent | — |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ventureforge.git
cd ventureforge

# Install dependencies
uv sync
# Or with pip: pip install -e .
```

### Configuration

Create a `.env` file (copy from `.env.example`):

```bash
# LLM Provider (choose one)
# Option 1: OpenAI-compatible (gpt-4o-mini, etc.)
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your_openai_key_here
LLM_MODEL=gpt-4o-mini

# Option 2: OpenRouter
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=your_openrouter_key_here
LLM_MODEL=anthropic/claude-3.5-sonnet

# Option 3: AMD vLLM (Qwen3.6)
LLM_BASE_URL=http://your-vllm-server:8000/v1
LLM_API_KEY=your_key_here
LLM_MODEL=Qwen/Qwen3.6-35B-A3B

# Optional (enhances pain point mining)
PRODUCTHUNT_API_KEY=your_key_here
YOUTUBE_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
```

### Run

```bash
# Gradio UI (recommended)
uv run app.py

# Or CLI
uv run python -m src.main --domain "developer tools" --output output.json
```

---

## ️ Tech Stack

| Layer | Tool |
|-------|------|
| Agent orchestration | LangGraph |
| LLM | OpenAI-compatible (gpt-4o-mini, Qwen3.6, etc.) |
| Validation | Pydantic v2 with structured output |
| State persistence | LangGraph MemorySaver (in-memory) |
| Web UI | Gradio |
| Data sources | Hacker News, Product Hunt, YouTube Data API |

---

##  API Keys

| Key | Where to Get | Required? |
|-----|-------------|-----------|
| LLM_BASE_URL / LLM_API_KEY | Your LLM provider (OpenAI, OpenRouter, vLLM) | ✅ Yes |
| PRODUCTHUNT_API_KEY | [producthunt.com](https://www.producthunt.com/v2/oauth/applications) | ⚠️ Optional |
| YOUTUBE_API_KEY | [Google Cloud Console](https://console.cloud.google.com/apis/credentials) | ⚠️ Optional |
| TAVILY_API_KEY | [tavily.com](https://tavily.com) | ⚠️ Optional |
| REDDIT_CLIENT_ID / SECRET | [reddit.com](https://www.reddit.com/prefs/apps) | ⚠️ Optional |
| HF_TOKEN | [huggingface.co](https://huggingface.co/settings/tokens) | ✅ Yes (AMD vLLM) |

---

## 🎯 Key Features

- **Binary Rubric Evaluation:** All subjective evaluation uses yes/no checks (Scorer: 8 checks, Critic: 5 checks)
- **Structured Output:** LLM responses use Pydantic v2 structured output for reliable schema validation
- **Reflection Loop:** Critic enforces quality via adversarial review with up to 2 revisions (configurable)
- **Evidence Validation:** Every pain point and pitch claim must cite a real source URL
- **Multi-Source Clustering:** Pain points clustered from multiple sources with 1-10 evidence items each
- **Provider Agnostic:** Works with any OpenAI-compatible LLM (gpt-4o-mini, Qwen3.6, Claude, etc.)

---

## 🧪 Testing

```bash
# Run component tests
pytest test/test_critic_component.py
pytest test/test_revision_feedback_flow.py

# Run end-to-end test (requires API keys)
pytest test/test_e2e.py
```

---

##  Documentation

- **[PROJECT.md](PROJECT.md)** - Detailed architecture and design decisions
- **[PROMPTS.md](PROMPTS.md)** - All agent prompts with Paul Graham framework
- **[orchestration.json](orchestration.json)** - Machine-readable orchestration spec

---

## 🤝 Contributing

Contributions are welcome! Fork the repository and submit a pull request.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- **AMD** for providing MI300X compute credits
- **Paul Graham** for the startup evaluation framework

---

**Built with ❤️ for AMD AI Hackathon | Track 1: AI Agents & Agentic Workflows**
