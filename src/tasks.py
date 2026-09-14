"""Sample wealth-management agent tasks for exercising the router.

These are illustrative multi-step tasks that mix trivial turns (which a small model
should handle) with harder reasoning turns (where a frontier model may be needed).
The point is to give the router a realistic mix so you can measure how many turns
actually escalate.

This is NOT investment advice and the data is synthetic. Swap in your own eval set —
ideally one with ground-truth answers so you can score accuracy, not just cost.

Each task has:
    id:           stable identifier
    difficulty:   rough expected tier ("easy" -> weak model should do; "hard" -> may escalate)
    prompt:       the user turn
    category:     coarse workload bucket
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    id: str
    difficulty: str  # "easy" | "medium" | "hard"
    category: str
    prompt: str


TASKS: list[Task] = [
    # --- Easy: lookups, formatting, restatement (weak model should handle) ---
    Task(
        id="wm-001",
        difficulty="easy",
        category="client_service",
        prompt="A client asks what the difference is between a Roth IRA and a "
        "traditional IRA in one short paragraph. Keep it plain-language.",
    ),
    Task(
        id="wm-002",
        difficulty="easy",
        category="reporting",
        prompt="Reformat this holdings list into a markdown table with columns "
        "Ticker, Shares, Asset Class: AAPL 120 equity; BND 300 fixed income; "
        "VNQ 75 real estate.",
    ),
    # Task(
    #     id="wm-003",
    #     difficulty="easy",
    #     category="client_service",
    #     prompt="Summarize this client note in two sentences: 'Client called, wants "
    #     "to increase monthly contributions from $500 to $750 starting next month "
    #     "and asked when the next review meeting is.'",
    # ),
    # --- Medium: multi-step but bounded ---
    Task(
        id="wm-004",
        difficulty="medium",
        category="planning",
        prompt="A 40-year-old client has $250k saved and contributes $1,500/month. "
        "Assuming a 6% nominal annual return, roughly what balance would they have "
        "at 65? Show the calculation steps.",
    ),
    Task(
        id="wm-005",
        difficulty="medium",
        category="portfolio_review",
        prompt="Given a portfolio that is 85% equities / 10% bonds / 5% cash for a "
        "client 3 years from retirement with a stated moderate risk tolerance, list "
        "the concerns and one rebalancing option. Note assumptions.",
    ),
    # Task(
    #     id="wm-006",
    #     difficulty="medium",
    #     category="compliance",
    #     prompt="Draft a suitability note explaining why a 70/30 allocation is "
    #     "appropriate for a client with a 15-year horizon and moderate risk tolerance. "
    #     "Flag anything a compliance reviewer would want documented.",
    # ),
    # --- Hard: reasoning under conflicting constraints (may escalate) ---
    Task(
        id="wm-007",
        difficulty="hard",
        category="planning",
        prompt="A client wants to retire in 8 years, fund two children's college in "
        "6 and 9 years, keep 12 months of expenses liquid, and minimize taxes. They "
        "have a taxable brokerage, a 401(k), and a 529. Propose a prioritized "
        "sequencing of contributions and withdrawals, explaining the tradeoffs and "
        "where the goals conflict.",
    ),
    # Task(
    #     id="wm-008",
    #     difficulty="hard",
    #     category="portfolio_review",
    #     prompt="Reconcile these conflicting client signals and recommend a path: the "
    #     "client says they are risk-averse, but they also insist on 90% equity exposure "
    #     "and are unhappy with any year of negative returns. Explain the contradiction, "
    #     "what you'd clarify, and a defensible allocation with rationale.",
    # ),
    # Task(
    #     id="wm-009",
    #     difficulty="hard",
    #     category="compliance",
    #     prompt="A client asks you to move their entire portfolio into a single "
    #     "concentrated position based on a hot tip. Walk through the fiduciary and "
    #     "suitability issues, what must be documented, and how to respond professionally "
    #     "while managing the relationship.",
    # ),
    # --- Very hard: dense multi-constraint quantitative reasoning ---
    # Designed to score a LOW p_solve so a capability-mode router escalates it to the
    # strong model. It requires after-tax return math, multi-account tax treatment,
    # RMDs, IRMAA thresholds, and Roth-conversion sequencing held together at once.
    Task(
        id="wm-010",
        difficulty="very_hard",
        category="planning",
        prompt=(
            "A 61-year-old couple (both retiring at 63) has $1.8M traditional IRA, "
            "$600k Roth, $900k taxable (cost basis $500k), and $120k cash. They want "
            "$140k/yr after-tax spending in today's dollars, indexed at 2.5% inflation, "
            "starting at 63. Social Security of $52k/yr combined begins at 70. Goals: "
            "(a) minimize lifetime federal tax, (b) stay under the second IRMAA tier "
            "each year once on Medicare at 65, (c) leave the Roth untouched as long as "
            "possible for heirs, and (d) never realize long-term gains at the 20% rate. "
            "Build a year-by-year withdrawal-and-Roth-conversion plan from 63 to 75, "
            "showing which account funds each year's spending, the conversion amount "
            "that fills the 24% bracket without breaching the IRMAA tier, how RMDs at 73 "
            "interact with the plan, and the assumptions and where goals (a)-(d) "
            "conflict. Show the bracket/IRMAA math for at least three representative years."
        ),
    ),
]


def as_messages(task: Task) -> list[dict[str, str]]:
    """Convert a Task into a Deep Agents / LangChain messages payload."""
    return [{"role": "user", "content": task.prompt}]


if __name__ == "__main__":
    from collections import Counter

    by_diff = Counter(t.difficulty for t in TASKS)
    print(f"{len(TASKS)} tasks: {dict(by_diff)}")
