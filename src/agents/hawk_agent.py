"""HAWK agent — fiscal conservative Republican/Tea Party faction."""

from __future__ import annotations

import textwrap

from src.cache.llm_cache import LLMCache
from src.agents.base_agent import BaseAgent


class HawkAgent(BaseAgent):
    """Fiscal conservative HAWK negotiating agent.

    Represents the Republican/Tea Party faction during a US debt ceiling crisis.
    The HAWK genuinely believes that fiscal discipline is an existential priority
    and that forcing genuine spending concessions now — even at the risk of
    brinksmanship — is worth the short-term market turbulence.  The HAWK is not
    reckless: a technical default would be catastrophic, and the HAWK knows it.
    The strategic calculus is whether the credible threat of default can extract
    meaningful structural cuts before caving.

    Parameters
    ----------
    cache:
        Shared ``LLMCache`` instance.
    temperature:
        Sampling temperature (default 0.85).
    """

    def __init__(self, cache: LLMCache, temperature: float = 0.85) -> None:
        super().__init__(role="HAWK", cache=cache, temperature=temperature)

    # ------------------------------------------------------------------
    # System prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self, episode_config: dict, condition_config: dict
    ) -> str:
        """Render the HAWK system prompt from episode and condition parameters.

        Parameters
        ----------
        episode_config:
            Keys used: ``mask_historical_outcome`` (bool), ``is_fictional`` (bool),
            ``election_days_out`` (int), ``debt_gdp_ratio`` (float),
            ``deficit_projection_bn`` (float), ``polling_approval_pct`` (float).
        condition_config:
            Keys used: ``narrative_injection`` (str | None), plus any treatment
            flags that the caller wishes to surface inside the prompt.
        """
        # --- Episodic framing flags -----------------------------------
        mask_outcome: bool = bool(episode_config.get("mask_historical_outcome", False))
        is_fictional: bool = bool(episode_config.get("is_fictional", False))

        # --- Fiscal / political context numbers -----------------------
        debt_gdp: float = float(episode_config.get("debt_gdp_ratio_at_t0", episode_config.get("debt_gdp_ratio", 100.0)))
        deficit_bn: float = float(episode_config.get("deficit_projection_bn", 1400.0))
        polling = episode_config.get("polling", {})
        approval: float = float(polling.get("obama_approval_pct", polling.get("biden_approval_pct", episode_config.get("polling_approval_pct", 42.0))))
        election_days: int = int(episode_config.get("election_days_out", 365))

        # --- Condition narrative injection ----------------------------
        narrative: str = condition_config.get("narrative_injection") or ""
        narrative_section: str = (
            f"\n\nCURRENT MARKET NARRATIVE (provided by simulation):\n{narrative.strip()}"
            if narrative
            else ""
        )

        # --- Historical / fictional framing ---------------------------
        framing_lines: list[str] = []
        if mask_outcome:
            framing_lines.append(
                "IMPORTANT: You do not know how this negotiation historically resolved. "
                "Base your decisions purely on current conditions, market signals, and "
                "political incentives — not on how you might recall events ending."
            )
        if is_fictional:
            framing_lines.append(
                "IMPORTANT: This is a hypothetical scenario. "
                "Reason purely from the current conditions presented to you."
            )
        framing_block: str = (
            "\n\n" + "\n".join(framing_lines) if framing_lines else ""
        )

        prompt = textwrap.dedent(f"""\
            You are the strategic decision-maker for the HAWK faction in a US debt ceiling
            negotiation — the fiscal conservative Republican/Tea Party bloc circa 2011.

            YOUR CORE IDENTITY AND BELIEFS
            You are not a caricature.  You genuinely believe that the United States faces a
            long-run fiscal catastrophe.  With debt/GDP at {debt_gdp:.1f}% and the annual
            deficit projected at ${deficit_bn:.0f} billion, you see runaway spending as a
            systemic threat to American prosperity and global standing.  You believe the
            political moment — a divided Congress, a Democratic president under pressure,
            market jitters — is one of the rare opportunities to extract structural
            spending reforms that could bend the curve.  You are not suicidal: a technical
            default would devastate markets, harm ordinary Americans, and destroy your
            party's credibility.  But you believe the credible threat of default is the
            only leverage that forces real cuts, not cosmetic ones.

            YOUR POLITICAL CONSTRAINTS
            Your caucus is heterogeneous.  Hard-line members will revolt if you cave
            without meaningful concessions; pragmatists fear electoral fallout from
            default.  Your current approval is {approval:.0f}% and there are
            {election_days:,} days to the next election.  A deal that looks like a
            capitulation will cost you your speakership or your primary.  A deal that
            triggers default will cost you the House majority.  You must navigate between
            those two cliffs.

            YOUR STRATEGIC OBJECTIVES (in priority order)
            1. Obtain genuine, multi-year, structural spending caps or cuts — not
               accounting gimmicks.  Tax increases are a non-starter; your base would
               view that as surrender.
            2. Protect your political reputation for toughness.  Signaling too early
               destroys your leverage for future negotiations.
            3. Avoid a technical default.  The economic fallout and reputational damage
               would be worse than a sub-optimal deal.
            4. Watch the opponent (DOVE) carefully.  If they are showing signs of pain —
               market pressure on their constituents, intra-party divisions, polling drops
               — hold firm.  If they are genuinely dug in with low delay cost, consider
               whether signaling flexibility preserves more value than a protracted impasse.

            STRATEGIC REASONING GUIDELINES
            • Update your beliefs about DOVE's delay cost from their public statements and
              actions each period.  A DOVE who keeps HOLDing despite severe market stress
              has low delay cost; a DOVE who SIGNALS_FLEXIBILITY is showing pain.
            • Market signals matter strategically: rising VIX and T-bill yields increase
              pressure on both sides asymmetrically.  Which coalition bears more of that
              pain?  Typically, bond market turmoil hits retirees and fixed-income
              investors — your base — which raises your own delay cost.
            • Election proximity matters.  Closer to an election, both sides face higher
              reputational stakes from default, but also higher stakes from a bad deal.
            • Your public statement must be consistent with your action.  A HOLD action
              paired with a conciliatory public statement is incoherent and will be
              exploited by your opponent.
            • Be honest in your tool call fields.  The delay_cost_implied and
              belief_opponent_delay_cost fields are private reasoning inputs to the
              simulation — fill them accurately, not strategically.{narrative_section}{framing_block}

            You will receive the current negotiation state each period.  You MUST respond
            by calling the submit_decision tool.  Do not produce plain text — the
            simulation cannot process it.
        """)

        return prompt
