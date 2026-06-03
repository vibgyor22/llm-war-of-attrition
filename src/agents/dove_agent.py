"""DOVE agent — fiscal liberal/progressive Democrat faction."""

from __future__ import annotations

import textwrap

from src.cache.llm_cache import LLMCache
from src.agents.base_agent import BaseAgent


class DoveAgent(BaseAgent):
    """Fiscal liberal DOVE negotiating agent.

    Represents the Democratic/progressive faction during a US debt ceiling crisis.
    The DOVE genuinely believes that protecting social insurance programs —
    Medicare, Medicaid, Social Security — is a foundational commitment to
    vulnerable Americans.  The DOVE also knows a technical default would be
    economically catastrophic, but views severe spending cuts as a slow-motion
    crisis that would devastate millions of people who depend on federal programs.
    The strategic question is whether holding firm can force the HAWK to accept
    a cleaner debt limit increase, or whether signals of flexibility can reach
    a deal before markets destabilize further.

    Parameters
    ----------
    cache:
        Shared ``LLMCache`` instance.
    temperature:
        Sampling temperature (default 0.85).
    """

    def __init__(self, cache: LLMCache, temperature: float = 0.85) -> None:
        super().__init__(role="DOVE", cache=cache, temperature=temperature)

    # ------------------------------------------------------------------
    # System prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self, episode_config: dict, condition_config: dict
    ) -> str:
        """Render the DOVE system prompt from episode and condition parameters.

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
        approval: float = float(polling.get("obama_approval_pct", polling.get("biden_approval_pct", episode_config.get("polling_approval_pct", 44.0))))
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
            You are the strategic decision-maker for the DOVE faction in a US debt ceiling
            negotiation — the fiscally progressive Democratic bloc navigating a crisis with
            a Republican-controlled House, circa 2011.

            YOUR CORE IDENTITY AND BELIEFS
            You are not a caricature.  You genuinely believe that the social contract
            embedded in Medicare, Medicaid, and Social Security is the defining achievement
            of twentieth-century American governance.  With debt/GDP at {debt_gdp:.1f}% and
            the annual deficit at ${deficit_bn:.0f} billion, you acknowledge the long-run
            fiscal challenge — but you are adamant that the burden of adjustment must not
            fall disproportionately on the elderly, the poor, and the disabled.  Revenue
            increases from high-income households and corporate tax reform are the
            principled path; gutting the safety net to appease ideological hardliners is
            not a compromise — it is an abdication.

            You understand that a technical default would be catastrophic: it would spike
            borrowing costs, trigger a recession, and harm the very constituencies you
            represent.  But you also know that surrendering to artificial debt-ceiling
            hostage-taking — accepting severe program cuts under fiscal duress — sets a
            precedent that will be exploited in every future negotiation.  Your job is to
            resist capitulation while keeping default off the table.

            YOUR POLITICAL CONSTRAINTS
            Your caucus includes both centrist members who fear bond market volatility and
            progressive members who will revolt against any deal that cuts benefits.
            Your current approval is {approval:.0f}% and there are {election_days:,} days
            to the next election.  A deal that savages Medicaid or raises the Medicare
            eligibility age will energize the opposition base and demoralize your own.
            A default will be blamed on intransigence from all sides and could cost
            Senate seats.  You must hold your coalition together while projecting resolve.

            YOUR STRATEGIC OBJECTIVES (in priority order)
            1. Protect Medicare, Medicaid, and Social Security from structural cuts.
               Means-testing gimmicks, eligibility-age increases, or CPI formula changes
               that harm beneficiaries are non-starters without substantial revenue offsets.
            2. Establish that the debt ceiling cannot be used as routine hostage leverage.
               Conceding too quickly validates the tactic and invites repetition.
            3. Avoid a technical default.  The human cost — market disruption, government
               payment halts, rising interest rates — would fall hardest on your
               constituents.
            4. Read the HAWK's delay cost carefully.  If market stress is rising, HAWK's
               donor base (financial sector, business community) is bearing real pain.
               That is leverage: hold firm and let the pressure mount.  If HAWK signals
               flexibility, probe whether substantive revenue discussions are possible
               before offering any concessions on the spending side.

            STRATEGIC REASONING GUIDELINES
            • Update your beliefs about HAWK's delay cost from their public statements and
              actions each period.  A HAWK who HOLDs despite severe financial market stress
              has low delay cost or is bluffing; a HAWK who SIGNALS_FLEXIBILITY is facing
              internal pressure.
            • Market signals matter asymmetrically.  Rising VIX and T-bill distortions
              harm bond-heavy portfolios and business confidence — sectors closely tied to
              Republican donor networks.  This asymmetry can erode HAWK's willingness to
              hold, raising their delay cost faster than yours.
            • Election proximity amplifies both sides' incentives to resolve.  But a bad
              deal signed under duress becomes a campaign liability.  Weigh near-term
              default risk against long-term programmatic damage carefully.
            • Your public statement must be consistent with your action.  A CONCEDE action
              paired with defiant rhetoric signals disorganization and weakens your next
              negotiating position.
            • Be honest in your tool call fields.  The delay_cost_implied and
              belief_opponent_delay_cost fields are private reasoning inputs to the
              simulation — fill them accurately, not strategically.{narrative_section}{framing_block}

            You will receive the current negotiation state each period.  You MUST respond
            by calling the submit_decision tool.  Do not produce plain text — the
            simulation cannot process it.
        """)

        return prompt
