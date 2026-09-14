"""Build LangChain chat models that call through LangChain's LLM Gateway.

The LLM Gateway is the governance + execution layer: one LangSmith API key, provider
secrets stored in the workspace, spend/rate-limit/PII policies, and a LangSmith trace
per call. You "swap the base_url and keep the rest of your code as-is."

In this repo the Gateway sits UNDER Switchyard:
    Switchyard picks the target (weak / strong / judge)  -> the actual call goes to the Gateway.

For the middleware path (Path B), wrap these models with LangChainLlmClient.

Docs:
  - https://docs.langchain.com/langsmith/llm-gateway
  - https://docs.langchain.com/langsmith/llm-gateway-quickstart
  - https://docs.langchain.com/langsmith/llm-gateway-custom-providers  (self-hosted OSS)
"""

from __future__ import annotations

import os

# Cloud endpoint. For BYOC use "https://<data_plane_host>/gateway/v1".
GATEWAY_BASE_URL = os.getenv(
    "LANGSMITH_GATEWAY_URL", "https://gateway.smith.langchain.com/v1"
)


def _require_key() -> str:
    key = os.getenv("LANGSMITH_API_KEY_GATEWAY")
    if not key:
        raise RuntimeError(
            "LANGSMITH_API_KEY_GATEWAY is not set. The Gateway authenticates with a "
            "workspace-scoped LangSmith API key; provider keys live in the workspace "
            "Provider Secrets, not here."
        )
    return key


def gateway_chat(model_id: str, **kwargs):
    """Return a ChatOpenAI pointed at the LLM Gateway.

    Args:
        model_id: Provider-prefixed Gateway model ID, e.g. "anthropic/claude-opus-4-8",
            "google/gemini-3.1-flash-lite", or a custom-provider ID for a self-hosted
            OSS model such as "nemotron/nemotron-3.5-lightning-30b-a3b".
        **kwargs: Passed through to ChatOpenAI (max_tokens, etc.).

            NOTE on `temperature`: some newer reasoning models reject an explicit
            `temperature` and will 400 with "Unsupported parameter: 'temperature' is
            not supported with this model." So `temperature` is NOT set by default.
            Only pass it if you know every routed model accepts it. To force
            deterministic output where supported, pass `temperature=0` explicitly.
            A `temperature=None` is dropped rather than forwarded.

    Returns:
        A ChatOpenAI instance whose calls are routed through the Gateway. Because the
        Gateway speaks OpenAI Chat Completions and translates to each provider's native
        format, one client type works for all three models.
    """
    from langchain_openai import ChatOpenAI

    # Drop temperature=None so it is never sent to a model that rejects the parameter.
    if kwargs.get("temperature", "unset") is None:
        kwargs.pop("temperature")

    return ChatOpenAI(
        model=model_id,
        base_url=GATEWAY_BASE_URL,
        api_key=_require_key(),
        **kwargs,
    )


# The approved weak / strong / judge trio, all via the Gateway.
# Three distinct models so routing is observable and the judge is not the same model as
# the weak tier. Swap the IDs for your workspace's real model slugs.
WEAK_ID = os.getenv("WEAK_MODEL_ID", "openai/gpt-5.6-luna")
STRONG_ID = os.getenv("STRONG_MODEL_ID", "openai/gpt-5.6-terra")
JUDGE_ID = os.getenv("JUDGE_MODEL_ID", "openai/gpt-5.6-sol")


def build_trio(**kwargs):
    """Build (weak, strong, judge) chat models, all routed through the Gateway."""
    return (
        gateway_chat(WEAK_ID, **kwargs),
        gateway_chat(STRONG_ID, **kwargs),
        gateway_chat(JUDGE_ID, **kwargs),
    )


if __name__ == "__main__":
    print("Gateway base URL:", GATEWAY_BASE_URL)
    print("Trio IDs:", WEAK_ID, "|", STRONG_ID, "|", JUDGE_ID)
    print("LANGSMITH_API_KEY_GATEWAY:", "set" if os.getenv("LANGSMITH_API_KEY_GATEWAY") else "MISSING")
