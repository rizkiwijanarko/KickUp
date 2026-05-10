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
2. **Clusters similar complaints** using LLM-based grouping (1-10 evidence sources per pain point)
3. **Generates startup ideas** that address multiple pain points
4. **Evaluates ideas** using binary yes/no rubrics (Paul Graham framework)
5. **Writes pitch briefs** with competitive intelligence and validation plans
6. **Critiques outputs** with adversarial review and reflection loop (max 3 revisions)

**Key Innovation:** All evaluation uses **binary yes/no checks** instead of subjective 0-1 scores, making results reproducible and auditable.

---

## 🏛️ Architecture

### Hierarchical + Reflection + Pressure-Test Pattern

```
                    ┌──────────────────┐
                    │   ORCHESTRATOR   │  ← Supervisor (routing only)
                    └────────┬─────────┘
                             │
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
        ┌────────┐      ┌────────┐      ┌────────┐
        │ Pain   │      │ Idea   │      │ Scorer │
        │ Point  │ ───► │ Gen    │ ───► │ (8 yes/│
        │ Miner  │      │        │      │ no chks│
        └────────┘      └────────┘      └────────┘
                             │
                             ▼
                        ┌────────┐      ┌──────────────────┐
                        │ Pitch  │      │   CRITIC AGENT   │
                        │ Writer │ ───► │  (5 yes/no chks) │
                        └────────┘      └────────┬─────────┘
                            │                    │
                   approve ─┴────────────────────┘
                   revise → loop back to target worker
```

### 6 Agents

| Agent | Role | Reflection Target? |
|-------|------|--------------------|
| **Orchestrator** | Routes tasks, manages state, handles reflection loop | — |
| **Pain Point Miner** | Scrapes HN, Product Hunt, YouTube; clusters complaints | ✅ |
| **Idea Generator** | Groups pain points into themes, generates ideas (one-at-a-time) | ✅ |
| **Scorer** | Evaluates ideas via 8 binary checks + fatal flaw detection | — |
| **Pitch Writer** | Writes one-page briefs (one-at-a-time, compressed prompts) | ✅ |
| **Critic** | Adversarial review with 5 core checks | — |

---

## 🔧 Token Optimization for vLLM 2048 Limit

The production vLLM server has a **~2048 token output limit**. We implemented three optimizations:

### 1. One-at-a-Time Generation

- **idea_generator**: Generates 1 idea per LLM call (instead of batch)
- **pitch_writer**: Generates 1 brief per LLM call (instead of batch)
- **Benefit**: Reduces output from 4,000+ tokens to ~2,000 tokens per call

### 2. Compressed Prompts

- **pitch_writer** uses compressed system prompt (1,123 → 502 words, 55% reduction)
- Preserves Paul Graham framework while saving ~800 input tokens

### 3. Filtered Pain Points

- Only sends relevant pain points (2-5 instead of 50)
- Limits evidence to top 2 items per pain point
- Truncates quotes to 300 chars
- **Benefit**: Reduces input tokens by ~50%

### Token Budget Per Call

| Agent | Input | Output | Total | Status |
|-------|-------|--------|-------|--------|
| idea_generator | 1,200 | 800 | 2,000 | ✅ Fits |
| pitch_writer | 1,000 | 2,000 | 3,000 | ⚠️ Over but works |
| scorer | 1,500 | 500 | 2,000 | ✅ Fits |
| critic | 1,800 | 400 | 2,200 | ⚠️ Occasional truncation |

**Result:** Pipeline runs reliably on all domains (healthcare, developer tools, finance, etc.)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- API keys (see below)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ventureforge.git
cd ventureforge

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Configuration

Create a `.env` file (copy from `.env.example`):

```bash
# Required for development
OPENROUTER_API_KEY=your_key_here

# Optional (enhances pain point mining)
PRODUCTHUNT_API_KEY=your_key_here
YOUTUBE_API_KEY=your_key_here

# Required for AMD vLLM production
LLM_BASE_URL=http://134.199.205.81:8000/v1
LLM_MODEL=Qwen/Qwen3.6-35B-A3B
```

### Run the Pipeline

#### Option 1: Gradio UI (Recommended)

```bash
uv run app.py
```

Then open http://127.0.0.1:7860 in your browser.

#### Option 2: CLI

```bash
uv run python -m src.main --domain "developer tools" --output output.json
```

---

## 📊 Example Output

### Pain Points (Clustered from 5+ sources)

```markdown
### 1. Healthcare Prior Authorization Denials (3 sources)
**Description:** Patients face insurance prior auth denials for necessary treatments
**Passes Rubric:** ✅ Yes

**Evidence:**
1. [Hacker News](https://news.ycombinator.com/item?id=45738250)
   > "My wife's cancer treatment was denied by insurance..."
2. [Product Hunt](https://www.producthunt.com/posts/...)
   > "Prior auth is a nightmare for chronic illness patients..."
```

### Scored Ideas (8 Binary Checks)

```json
{
  "title": "SecondOpinion Advocate",
  "verdict": "pursue",
  "yes_count": 7,
  "feasibility_rubric": {
    "can_be_solved_manually_first": true,
    "has_schlep_or_unsexy_advantage": true,
    "can_2_3_person_team_build_mvp_in_6_months": true
  },
  "demand_rubric": {
    "addresses_at_least_2_pain_points": true,
    "is_painkiller_not_vitamin": true,
    "has_clear_vein_of_early_adopters": true
  },
  "novelty_rubric": {
    "differentiated_from_current_behavior": true,
    "has_path_out_of_niche": false
  },
  "fatal_flaws": [],
  "core_assumption": "Patients will pay for specialized prior auth advocacy",
  "one_risk": "Insurance companies may block or delay responses"
}
```

### Pitch Brief (Investor-Ready)

```markdown
# SecondOpinion Advocate

**Tagline:** AI-powered prior authorization advocate for chronic illness patients

## Problem
Patients with chronic conditions face insurance prior auth denials...

## Solution
SecondOpinion Advocate is an AI-powered service that...

## Target User
Patients in r/chronicillness who have recently posted about prior auth denials

## Competitive Landscape
**Current behavior:** Patients call insurance companies themselves (slow, frustrating)
**Incumbents:** Hospital financial counselors (expensive, slow)
**Our thesis:** General AI tools lack nuance for specific insurance plan rules...

## Evidence
- https://news.ycombinator.com/item?id=45738250
- https://www.producthunt.com/posts/...
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| Agent orchestration | LangGraph |
| LLM (dev) | OpenRouter (any OpenAI-compatible) |
| LLM (production) | Qwen/Qwen3.6-35B-A3B via vLLM on AMD MI300X |
| Validation | Pydantic v2 |
| State persistence | LangGraph SQLite checkpointer |
| Web UI | Gradio |
| Data sources | Hacker News, Product Hunt, YouTube Data API |
| Token optimization | One-at-a-time generation + compressed prompts |

---

## 📁 Project Structure

```
ventureforge/
├── README.md                    # This file
├── PROJECT.md                   # Detailed architecture docs
├── PROMPTS.md                   # Agent prompts
├── orchestration.json           # Machine-readable spec
├── pyproject.toml
├── .env.example
│
├── src/
│   ├── main.py                  # CLI entry
│   ├── config.py                # Pydantic settings
│   ├── state/schema.py          # All Pydantic models
│   ├── graph.py                 # LangGraph orchestrator
│   │
│   ├── agents/
│   │   ├── pain_point_miner.py
│   │   ├── idea_generator.py   # ✨ One-at-a-time generation
│   │   ├── scorer.py
│   │   ├── pitch_writer.py     # ✨ One-at-a-time + compressed prompts
│   │   └── critic.py
│   │
│   ├── tools/
│   │   ├── hackernews_scraper.py
│   │   ├── producthunt_scraper.py
│   │   └── youtube_scraper.py
│   │
│   └── llm/
│       ├── client.py            # OpenAI-compatible factory
│       └── prompts.py
│
├── agent_prompts/
│   └── pitch_writer_prompt_compressed.txt  # ✨ Optimized prompt
│
└── app.py                       # Gradio UI
```

---

## 🔑 Required API Keys

| Key | Where to Get | Free? | Required? |
|-----|-------------|-------|-----------|
| OPENROUTER_API_KEY | [openrouter.ai](https://openrouter.ai) | Pay-per-token | ✅ Yes (dev) |
| PRODUCTHUNT_API_KEY | [producthunt.com/v2/oauth/applications](https://www.producthunt.com/v2/oauth/applications) | ✅ Free | ⚠️ Optional |
| YOUTUBE_API_KEY | [console.cloud.google.com/apis/credentials](https://console.cloud.google.com/apis/credentials) | ✅ Free (10K units/day) | ⚠️ Optional |
| HF_TOKEN | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | ✅ Free | ✅ Yes (AMD) |

**Note:** Reddit API requires application approval and is currently disabled.

---

## 🎯 Key Features

### 1. Binary Rubric Evaluation

All subjective evaluation uses **yes/no checks** instead of 0-1 scores:

**Scorer (8 checks):**
- Feasibility: manual-first, schlep advantage, 2-3 person MVP
- Demand: addresses 2+ pain points, painkiller not vitamin, clear early adopters
- Novelty: differentiated, path out of niche

**Critic (5 checks):**
- All claims evidence-backed
- No hallucinated source URLs
- Tagline under 12 words
- Target is contained fire (specific community)
- Competition embraced with thesis

### 2. Reflection Loop

The Critic enforces quality via adversarial review:

```python
if critique.all_pass:
    return END

elif state.revision_counts[idea_id] < max_revisions:
    # Loop back to the worker the critic targeted
    return critique.target_agent  # pain_point_miner | idea_generator | pitch_writer

else:
    return END  # max revisions reached, ship as-is
```

### 3. Evidence Validation

Every pain point and pitch claim must cite a **real source URL**. The Critic verifies:
- No hallucinated URLs
- All evidence links appear in pain_points evidence arrays
- Verbatim quotes from sources

### 4. Multi-Source Pain Point Clustering

Pain points are clustered from multiple sources:
- Each pain point has 1-10 evidence items
- LLM groups similar complaints
- Prefers clusters with 2+ sources over single-source complaints

---

## 🧪 Testing

### Run Full Pipeline Test

```bash
# Test with healthcare domain (most challenging)
uv run python -m src.main --domain "healthcare" --max-pain-points 5 --output test.json

# Check results
cat test.json | python3 -c "import sys, json; data=json.load(sys.stdin); print(f'Stage: {data[\"current_stage\"]}'); print(f'Ideas: {len(data[\"ideas\"])}'); print(f'Pitch briefs: {len(data[\"pitch_briefs\"])}')"
```

### Expected Output

```
Stage: completed
Ideas: 2
Pitch briefs: 2
```

### Run Component Tests

```bash
# Test individual agents
pytest test/test_critic_component.py
pytest test/test_revision_feedback_flow.py

# Test end-to-end (requires API keys)
pytest test/test_e2e.py
```

---

## 🚀 AMD Developer Cloud Deployment

### Setup

1. **Create GPU Droplet** (MI300X)
2. **Use vLLM Quick Start image** (pre-built ROCm + vLLM + PyTorch)
3. **Serve model:**

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --max-num-seqs 4
```

4. **Update .env:**

```bash
LLM_BASE_URL=http://your-amd-instance-ip:8000/v1
LLM_MODEL=Qwen/Qwen3.6-35B-A3B
```

5. **Run pipeline:**

```bash
uv run app.py
```

---

## 📊 Performance

### Agent Timings (Healthcare Domain)

| Agent | Duration | Notes |
|-------|----------|-------|
| pain_point_miner | 75.5s | Scrapes 3 sources in parallel |
| idea_generator | 5.6s | 2 ideas, one-at-a-time |
| scorer | 9.8s | Evaluates 2 ideas |
| pitch_writer | 26.2s | 2 briefs, one-at-a-time |
| critic | 7.3s | Reviews 2 briefs |
| **Total** | **124.5s** | **~2 minutes** |

---

## 🎯 Success Criteria

- [x] One command runs all 6 agents end-to-end without crashing
- [x] Critic reflection loop triggers at least once during demo run
- [x] Every pain point has a real source URL (no hallucinations)
- [x] At least one "park" verdict appears due to a fatal flaw
- [x] Gradio UI shows live agent execution with rubric outcomes
- [x] Final demo runs on AMD vLLM (Qwen 3.6 35B)
- [x] Pipeline works reliably on all domains (healthcare, developer tools, finance)
- [x] Token optimization enables consistent operation within vLLM 2048 limit

---

## 📝 Documentation

- **[PROJECT.md](PROJECT.md)** - Detailed architecture and design decisions
- **[PROMPTS.md](PROMPTS.md)** - All agent prompts with Paul Graham framework
- **[orchestration.json](orchestration.json)** - Machine-readable orchestration spec
- **[FINAL_TEST_RESULTS.md](FINAL_TEST_RESULTS.md)** - Test results and validation

---

## 🤝 Contributing

This project was built for the AMD AI Hackathon. Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- **AMD** for providing MI300X compute credits
- **LangChain/LangGraph** for the agent orchestration framework
- **Paul Graham** for the startup evaluation framework
- **Hacker News, Product Hunt, YouTube** for community data sources

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ for AMD AI Hackathon | Track 1: AI Agents & Agentic Workflows**
