# financial-services-routing

Testing **NVIDIA NeMo Switchyard** for LLM model routing, applied to a wealth-management
agent workload.

This repo reproduces and adapts the approach from the LangChain benchmark
["How many of your agent's calls actually need a frontier model?"](https://www.langchain.com/blog/switchyard-agent-routing-benchmark)
(NVIDIA NeMo Switchyard + Deep Agents). The core question: **on a wealth-management
agent workload, how many turns actually need a frontier model, and what does routing
the rest to a smaller model cost you in accuracy?**

## What's here

| Path | Purpose |
| --- | --- |
| `notebooks/switchyard_routing.ipynb` | The main walkthrough — install, configure, run both routing paths, route through the LLM Gateway, and compute the cost tradeoff |
| `config/routes.toml` | Switchyard **server** config in escalation mode, calling providers **directly** |
| `config/routes.gateway.toml` | Same server config, but all 3 models routed through **LangChain's LLM Gateway** |
| `src/gateway_models.py` | Helper to build LangChain chat models that call through the LLM Gateway |
| `src/tasks.py` | Small wealth-management agent task set used to exercise the router |
| `src/cost_formula.py` | The `minimum offload = judge cost / (expensive cost - cheap cost)` calculator |
| `requirements.txt` | Python dependencies |

## Two ways to run Switchyard (they are not the same)

This is the single most important thing to understand before running anything.

### 1. Server, escalation mode (reproduces the benchmark numbers)

- Route `type = "llm_classifier"` with `mode = "escalation"` in a server `routes.toml`.
- Every turn starts on the **weak** model. A separate **judge** model reads each completed
  turn and votes on whether the agent is on track. **Two consecutive negative verdicts**
  (`confirmations = 2`) permanently move that task to the **strong** model ("two strikes,
  one-way door").
- The judge runs on every weak-tier turn, so it is a fixed cost tax — in the benchmark it
  was **21.2%** of routed spend.
- Switchyard core is a **Rust** proxy. You `cargo install switchyard-server`, point your
  agent's `base_url` at it, and it translates between OpenAI Chat / Anthropic Messages / etc.
- **This is the path that produced the benchmark's 74%-cheaper / ~6-points-less-accurate result.**

### 2. In-process middleware (`SwitchyardRoutingMiddleware`)

- Clone `Switchyard` + `langchain-nvidia`, `pip install -e` both, use the `libsy` Python
  bindings inside a Deep Agent. No separate service.
- Exposes `stage_router` (signal-driven, no judge, ~0 added latency), an LLM task
  classifier, random, and no-op.
- The benchmark explicitly notes the middleware **will not reproduce the escalation numbers
  directly** — `stage_router` is a different algorithm that keys off tool-call / tool-result
  signals rather than a judge's verdict.
- Every routed `AIMessage` carries the trace in `response_metadata["switchyard"]`
  (`selected_model` + `decisions`).

**Bottom line:** use the **server** to reproduce the benchmark, use the **middleware** for
the cleanest in-process integration. This repo covers both.

## Routing all 3 models through LangChain's LLM Gateway

You can put **LangChain's LLM Gateway** underneath the router so every model call (weak,
strong, and judge) flows through one governed endpoint. Two layers, two jobs:

```
agent ──> Switchyard (routes) ──> LLM Gateway (governs + calls) ──> Anthropic / Gemini / Nemotron
```

- **Switchyard** owns the **routing decision** (weak vs strong vs judge, escalation).
- **LLM Gateway** owns **execution + governance**: one LangSmith API key (provider keys
  live in workspace Provider Secrets), spend / rate-limit / PII policies, and a LangSmith
  trace per call — applied no matter which tier the router picked.

Wiring:
- **Server path:** use `config/routes.gateway.toml` (points every `llm_client` at the
  Gateway base URL, authenticates with `LANGSMITH_API_KEY_GATEWAY`, uses provider-prefixed Gateway
  model IDs). The escalation route is unchanged.
- **Middleware path:** build each model with `src/gateway_models.py` (`base_url` = Gateway,
  `api_key` = LangSmith key) and hand them to `LangChainLlmClient`.

One-time admin setup in the LangSmith UI: enable the Gateway, add Provider Secrets (add a
self-hosted Nemotron as a *custom provider*), grant your API key access. See section 5 of
the notebook and the [LLM Gateway docs](https://docs.langchain.com/langsmith/llm-gateway).

## The cost formula (run this before you build anything)

```
minimum offload = judge cost / (expensive cost - cheap cost)
```

The judge is a fixed tax on every run. The question is never "is my cheap model good
enough?" — it is "is the price gap between my two models wide enough to pay for the judge?"
If your two models are close in price, this number climbs past 100% and routing cannot pay
unless you host the cheap model yourself. See `src/cost_formula.py` and the notebook.

## Prerequisites

- Python **3.12+** (required by the middleware path)
- [Rust + Cargo](https://rust-lang.org/tools/install/) (for the server path)
- API keys for the providers you route between (see `.env.example`)

## Quickstart

### With uv (recommended)

```bash
uv sync                 # creates .venv and installs deps from pyproject.toml / uv.lock
cp .env.example .env    # fill in your keys
uv run jupyter lab notebooks/switchyard_routing.ipynb
```

`requirements.txt` is kept in sync for pip users, but `pyproject.toml` + `uv.lock` are the
source of truth. Use `uv sync --group dev` to also install notebook-authoring tools.

### With pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your keys
jupyter lab notebooks/switchyard_routing.ipynb
```

> The Switchyard **middleware** and **server** are still installed separately (from source
> and via Cargo respectively) under either workflow — see the notes in `pyproject.toml`.

## A note on models

The benchmark paired **Nemotron 3.5 Lightning (30B)** (weak) with **Claude Opus 4.8**
(strong) and **Gemini 3.1 Flash Lite** (judge). Swap in whatever weak/strong/judge trio
your desk has approved — the config and formula are model-agnostic. Keep the pairing
**lopsided in price** or the formula will tell you routing can't pay.

## Disclaimers

- Switchyard is **pre-alpha** ("Demo server, not for production use"; `libsy` is Beta). APIs
  change fast. Pin versions.
- The benchmark suite was **saturated** (only ~8 points separated the 30B model from the
  frontier model), which gives routing little room to prove value. Treat all numbers as a
  measurement of *one* workload. Run your own.
- This is research / evaluation tooling, not investment advice, and the sample tasks are
  illustrative only.
