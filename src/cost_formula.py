"""Cost tradeoff calculator for LLM model routing.

Implements the formula from the LangChain / NVIDIA Switchyard benchmark:

    minimum offload = judge cost / (expensive cost - cheap cost)

The judge is a fixed tax on every run (it reads every weak-tier turn until a task
escalates). So the routing question is never "is my cheap model good enough?" but
"is the price gap between my two models wide enough to pay for the judge?"

If the required offload exceeds 1.0 (100%), routing cannot pay against the strong
model alone unless you shrink the judge cost or host the cheap model locally
(near-zero inference cost widens the gap).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingEconomics:
    """Result of a routing cost-tradeoff analysis."""

    minimum_offload: float
    """Fraction of turns that must go to the cheap model for routing to pay
    (relative to running the expensive model alone). May exceed 1.0 (impossible)."""

    feasible: bool
    """True when minimum_offload <= 1.0, i.e. routing *can* pay in principle."""

    price_gap: float
    """expensive_cost - cheap_cost, per turn/run in the same unit as inputs."""

    def summary(self) -> str:
        pct = self.minimum_offload * 100
        if not self.feasible:
            return (
                f"Minimum offload = {pct:.1f}% (> 100%): routing CANNOT pay against "
                f"the strong model alone. The price gap ({self.price_gap:.6f}) is too "
                f"narrow to cover the judge. Shrink the judge or self-host the cheap model."
            )
        return (
            f"Minimum offload = {pct:.1f}%: route at least this share of turns to the "
            f"cheap model for routing to beat the strong model alone."
        )


def minimum_offload(
    judge_cost: float,
    expensive_cost: float,
    cheap_cost: float,
) -> RoutingEconomics:
    """Compute the minimum fraction of turns to offload for routing to pay.

    Args:
        judge_cost: Fixed judge cost per run (same unit as the model costs below).
        expensive_cost: Cost of one turn on the strong/frontier model.
        cheap_cost: Cost of one turn on the weak/cheap model.

    Returns:
        RoutingEconomics with the minimum offload fraction and feasibility flag.

    Raises:
        ValueError: If the price gap is not positive (expensive must cost more).
    """
    price_gap = expensive_cost - cheap_cost
    if price_gap <= 0:
        raise ValueError(
            "expensive_cost must be strictly greater than cheap_cost; "
            f"got expensive={expensive_cost}, cheap={cheap_cost}."
        )
    required = judge_cost / price_gap
    return RoutingEconomics(
        minimum_offload=required,
        feasible=required <= 1.0,
        price_gap=price_gap,
    )


if __name__ == "__main__":
    # Benchmark's own numbers: judge $0.64/run, price gap $10.73 -> needed ~5.9% offload.
    econ = minimum_offload(judge_cost=0.64, expensive_cost=0.72 + 10.73, cheap_cost=0.72)
    print(econ.summary())
    print("(Benchmark reported ~5.9%; they offloaded 93%, clearing the bar ~16x.)")
