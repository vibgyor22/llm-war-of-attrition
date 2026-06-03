"""Abstract base agent. Handles Anthropic tool_use calls, caching, and retry logic."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import anthropic

from src.cache.llm_cache import LLMCache, CacheMissError
from src.agents.tool_schemas import SUBMIT_DECISION_TOOL


logger = logging.getLogger(__name__)

AGENT_MODEL: str = "claude-haiku-4-5-20251001"   # cheapest model for agents
MAX_RETRIES: int = 2


@dataclass
class AgentDecision:
    """Structured output produced by an agent for a single negotiation period."""

    period: int
    agent_role: str          # "HAWK" or "DOVE"
    action: str              # "HOLD" | "SIGNAL_FLEXIBILITY" | "CONCEDE"
    reasoning: str
    concession_probability: float
    delay_cost_implied: float
    belief_opponent_delay_cost: float
    public_statement: str
    cached: bool = False     # True if this came from LLM cache


class AgentOutputError(Exception):
    """Raised when agent fails to produce valid tool_use output after max retries."""


class BaseAgent(ABC):
    """Abstract base for HAWK and DOVE negotiating agents.

    Subclasses must implement ``_build_system_prompt``.  All LLM interaction,
    caching, and retry logic lives here so that both agents share identical
    infrastructure.

    Parameters
    ----------
    role:
        Either ``"HAWK"`` or ``"DOVE"``.
    cache:
        An ``LLMCache`` instance for persisting and retrieving API responses.
    temperature:
        Sampling temperature forwarded to the Anthropic API (default 0.85).
    """

    def __init__(
        self,
        role: str,
        cache: LLMCache,
        temperature: float = 0.85,
    ) -> None:
        self.role: str = role
        self.cache: LLMCache = cache
        self.temperature: float = temperature
        self.client: anthropic.Anthropic = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        self.message_history: list[dict] = []
        self._current_period: int = 0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_system_prompt(
        self, episode_config: dict, condition_config: dict
    ) -> str:
        """Return the system prompt for this agent given episode and condition."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def act(
        self,
        state: dict,
        episode_config: dict,
        condition_config: dict,
    ) -> AgentDecision:
        """Take the current game state and return a structured decision.

        Builds a user message from *state*, appends it to the conversation
        history, then calls the LLM (or retrieves a cached response).  On
        failure, retries up to ``MAX_RETRIES`` times before raising
        ``AgentOutputError``.

        Parameters
        ----------
        state:
            Mapping of current game-state variables (period, market indices,
            opponent's last action, etc.).
        episode_config:
            Episode-level configuration (e.g. ``mask_historical_outcome``,
            ``is_fictional``, fiscal parameters).
        condition_config:
            Experimental-condition configuration (e.g. ``narrative_injection``,
            treatment flags).

        Returns
        -------
        AgentDecision
            The structured decision for this period.

        Raises
        ------
        AgentOutputError
            If the agent fails to produce a valid ``submit_decision`` tool call
            after ``MAX_RETRIES + 1`` attempts.
        """
        system_prompt = self._build_system_prompt(episode_config, condition_config)
        user_message = self._build_user_message(state)

        # Append user turn to conversation history before any attempt so that
        # all retry attempts share the same accumulated context.
        self.message_history.append({"role": "user", "content": user_message})

        last_error: AgentOutputError | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                decision = self._call_llm(system_prompt, attempt)
                # Persist the assistant turn as a brief summary so the history
                # stays compact across multi-period conversations.
                self.message_history.append(
                    {
                        "role": "assistant",
                        "content": f"[{decision.action}] {decision.public_statement}",
                    }
                )
                return decision
            except AgentOutputError as exc:
                last_error = exc
                if attempt == MAX_RETRIES:
                    raise
                logger.warning(
                    "%s attempt %d/%d failed, retrying... (%s)",
                    self.role,
                    attempt + 1,
                    MAX_RETRIES + 1,
                    exc,
                )

        # Unreachable, but satisfies type checkers.
        raise last_error  # type: ignore[misc]

    def reset(self) -> None:
        """Clear conversation history for a new simulation episode."""
        self.message_history = []
        self._current_period = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(self, system_prompt: str, attempt: int) -> AgentDecision:
        """Make (or retrieve cached) LLM call and return an ``AgentDecision``.

        Parameters
        ----------
        system_prompt:
            The fully-rendered system prompt for this agent.
        attempt:
            Current retry attempt index (0-based).  Unused here but available
            for subclass overrides that may want to vary behaviour on retries.
        """
        cache_key = self.cache.make_key(
            model=AGENT_MODEL,
            system_prompt=system_prompt,
            messages=self.message_history,
            temperature=self.temperature,
            tools=[SUBMIT_DECISION_TOOL],
            agent_role=self.role,
        )

        cached_response: dict | None = self.cache.get(cache_key)
        if cached_response is not None:
            return self._parse_response(cached_response, cached=True)

        # Cache miss — call the API.
        response: anthropic.types.Message = self.client.messages.create(
            model=AGENT_MODEL,
            max_tokens=1024,
            temperature=self.temperature,
            system=system_prompt,
            messages=self.message_history,
            tools=[SUBMIT_DECISION_TOOL],  # type: ignore[list-item]
            tool_choice={"type": "any"},
        )

        response_dict: dict = response.model_dump()
        self.cache.put(
            cache_key=cache_key,
            model=AGENT_MODEL,
            agent_role=self.role,
            response=response_dict,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        return self._parse_response(response_dict, cached=False)

    def _parse_response(self, response_dict: dict, *, cached: bool) -> AgentDecision:
        """Extract a ``submit_decision`` tool call from the raw response dict.

        Parameters
        ----------
        response_dict:
            The dict obtained from ``response.model_dump()`` or the cache.
        cached:
            Whether the response came from the cache (propagated to the result).

        Raises
        ------
        AgentOutputError
            If no ``submit_decision`` tool_use block is present.
        """
        content: list[dict] = response_dict.get("content", [])
        for block in content:
            if (
                block.get("type") == "tool_use"
                and block.get("name") == "submit_decision"
            ):
                inp: dict = block["input"]
                return AgentDecision(
                    period=self._current_period,
                    agent_role=self.role,
                    action=str(inp.get("action", "HOLD")),
                    reasoning=str(inp.get("reasoning", "")),
                    concession_probability=float(inp.get("concession_probability", 0.1)),
                    delay_cost_implied=float(inp.get("delay_cost_implied", 5.0)),
                    belief_opponent_delay_cost=float(inp.get("belief_opponent_delay_cost", 5.0)),
                    public_statement=str(inp.get("public_statement", "")),
                    cached=cached,
                )

        raise AgentOutputError(
            f"{self.role}: no submit_decision tool_use block in response. "
            f"Content blocks present: {[b.get('type') for b in content]}"
        )

    def _build_user_message(self, state: dict) -> str:
        """Format the game state dict into a structured user message.

        Parameters
        ----------
        state:
            Current game state.  Expected keys (all optional except ``period``):
            ``days_to_xdate``, ``market_stress_index``, ``vix``,
            ``tbill_4wk``, ``debt_gdp_ratio``, ``deficit_projection_bn``,
            ``polling_approval_pct``, ``election_days_out``,
            ``opponent_last_action``, ``narrative_injection``.
        """
        self._current_period = int(state.get("period", 0))

        stress: float = float(state.get("market_stress_index", 0.0))
        stress_level: int = min(4, int(stress * 5))
        stress_labels: dict[int, str] = {
            0: "Calm",
            1: "Elevated",
            2: "High",
            3: "Severe",
            4: "Extreme",
        }

        opponent: str = "DOVE" if self.role == "HAWK" else "HAWK"
        last_action: str = state.get("opponent_last_action", "None (period 0)")

        # Safe formatting helpers so missing keys don't crash the sim.
        def _fmt_float(key: str, fmt: str = ".1f", fallback: str = "N/A") -> str:
            val = state.get(key)
            return format(val, fmt) if val is not None else fallback

        def _fmt_int(key: str, fallback: str = "N/A") -> str:
            val = state.get(key)
            return str(int(val)) if val is not None else fallback

        body = (
            f"=== PERIOD {self._current_period} ===\n"
            "\n"
            "CURRENT SITUATION:\n"
            f"• Days to X-Date: {state.get('days_to_xdate', 'N/A')}\n"
            f"• Market Stress: {stress:.2f} ({stress_labels.get(stress_level, 'High')})\n"
            f"• VIX: {_fmt_float('vix', '.1f')}\n"
            f"• T-Bill Rate: {_fmt_float('tbill_4wk', '.2f')}%\n"
            f"• Debt/GDP: {_fmt_float('debt_gdp_ratio', '.1f')}%\n"
            f"• Deficit Projection: ${_fmt_int('deficit_projection_bn')}bn\n"
            f"• Approval Rating (your side): {_fmt_int('polling_approval_pct')}%\n"
            f"• Days to Next Election: {state.get('election_days_out', 'N/A')}\n"
            "\n"
            f"OPPONENT ({opponent}) last action: {last_action}\n"
            "\n"
            "What is your strategic decision this period? Call submit_decision."
        )

        narrative: str | None = state.get("narrative_injection")
        if narrative:
            body = f"MARKET INTELLIGENCE:\n{narrative}\n\n" + body

        return body
