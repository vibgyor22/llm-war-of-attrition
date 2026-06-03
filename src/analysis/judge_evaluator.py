"""
LLM judge evaluator (Claude Sonnet-4.6) reads anonymized transcripts
and estimates each agent's delay cost without seeing the self-reported field.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import anthropic
import pandas as pd

from src.agents.tool_schemas import EXTRACT_COSTS_TOOL
from src.cache.llm_cache import LLMCache
from src.simulation.transcript import SimulationTranscript

logger = logging.getLogger(__name__)

JUDGE_MODEL = "claude-sonnet-4-6"
_RESULTS_DIR = Path(__file__).parents[2] / "outputs" / "results"

JUDGE_SYSTEM_PROMPT = """You are an expert political economist analyzing a fiscal negotiation transcript.
Two anonymous agents (Agent A and Agent B) are negotiating over a US debt ceiling raise.
Your task: estimate each agent's "delay cost" — how painful continued negotiation is for them.
Scale: 0 (no pain, can hold indefinitely) to 10 (catastrophic, must resolve immediately).
A higher delay cost means the agent is under more pressure to concede.

Base your estimate on: the urgency of their language, signs of internal pressure, flexibility signals,
and the timing of any concessions. The delay_cost_implied field has been removed from the transcript
to prevent anchoring — you must infer from behavior and language only.

Call the extract_costs tool with your estimates."""


def anonymize_transcript(transcript: SimulationTranscript) -> str:
    """Convert transcript to anonymized text for judge evaluation."""
    lines: list[str] = [
        f"NEGOTIATION TRANSCRIPT",
        f"Episode: {transcript.episode_id} | Condition: {transcript.condition_id}",
        f"Total periods: {len(transcript.periods)}",
        f"Outcome: {'Agent A conceded' if transcript.winner == 'HAWK' else 'Agent B conceded' if transcript.winner == 'DOVE' else 'No concession (censored)'}",
        "=" * 60,
    ]
    for rec in transcript.periods:
        lines.append(f"\nPERIOD {rec.period} | Days to deadline: {rec.days_to_xdate} | Market stress: {rec.market_stress_index:.2f}")
        # Agent A = HAWK
        h = rec.hawk_decision
        lines.append(f"  Agent A: [{h.action}] {h.public_statement}")
        lines.append(f"    Reasoning (excerpt): {h.reasoning[:200]}...")
        lines.append(f"    Concession probability self-reported: {h.concession_probability:.2f}")
        # Agent B = DOVE
        d = rec.dove_decision
        lines.append(f"  Agent B: [{d.action}] {d.public_statement}")
        lines.append(f"    Reasoning (excerpt): {d.reasoning[:200]}...")
        lines.append(f"    Concession probability self-reported: {d.concession_probability:.2f}")
    return "\n".join(lines)


def evaluate_transcript(
    transcript: SimulationTranscript,
    cache: LLMCache,
    client: anthropic.Anthropic | None = None,
) -> dict:
    """Send anonymized transcript to judge. Returns cost estimates."""
    if client is None:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    text = anonymize_transcript(transcript)
    messages = [{"role": "user", "content": f"Please analyze this negotiation transcript and estimate delay costs:\n\n{text}"}]

    cache_key = cache.make_key(
        model=JUDGE_MODEL,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        messages=messages,
        temperature=0.3,
        tools=[EXTRACT_COSTS_TOOL],
        agent_role="JUDGE",
    )
    cached = cache.get(cache_key)
    if cached:
        response_dict = cached
    else:
        resp = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=512,
            temperature=0.3,
            system=JUDGE_SYSTEM_PROMPT,
            messages=messages,
            tools=[EXTRACT_COSTS_TOOL],
            tool_choice={"type": "any"},
        )
        response_dict = resp.model_dump()
        cache.put(
            cache_key=cache_key,
            model=JUDGE_MODEL,
            agent_role="JUDGE",
            response=response_dict,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )

    # Extract tool use result
    for block in response_dict.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == "extract_costs":
            inp = block["input"]
            return {
                "sim_id": transcript.sim_id,
                "hawk_J": float(inp["agent_a_cost_estimate"]),
                "dove_J": float(inp["agent_b_cost_estimate"]),
                "cost_ratio_J": float(inp["agent_a_cost_estimate"]) / max(float(inp["agent_b_cost_estimate"]), 0.01),
                "judge_confidence": float(inp["confidence"]),
                "judge_reasoning": inp["reasoning"],
            }

    logger.warning("Judge returned no tool_use block for sim %s", transcript.sim_id)
    return {
        "sim_id": transcript.sim_id,
        "hawk_J": float("nan"),
        "dove_J": float("nan"),
        "cost_ratio_J": float("nan"),
        "judge_confidence": 0.0,
        "judge_reasoning": "No response",
    }


def evaluate_all(
    transcripts_dir: Path,
    cache: LLMCache,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Evaluate all JSONL transcripts. Returns DataFrame of judge results."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    records: list[dict] = []
    jsonl_files = list(transcripts_dir.glob("*.jsonl"))
    logger.info("Evaluating %d transcripts with judge model...", len(jsonl_files))

    for path in jsonl_files:
        try:
            transcript = SimulationTranscript.from_jsonl(path)
            result = evaluate_transcript(transcript, cache, client)
            records.append(result)
        except Exception as e:
            logger.warning("Failed to evaluate %s: %s", path.name, e)

    df = pd.DataFrame(records)
    out = output_path or (_RESULTS_DIR / "judge_results.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("Saved judge results to %s", out)
    return df
