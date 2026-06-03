"""Anthropic tool_use schemas for agent decisions in the debt-ceiling negotiation simulation."""

SUBMIT_DECISION_TOOL: dict = {
    "name": "submit_decision",
    "description": (
        "Submit your strategic decision for this negotiation period. "
        "You MUST call this tool — do not respond with plain text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Your private strategic reasoning (not shown publicly). 2-4 sentences max.",
                "maxLength": 600,
            },
            "action": {
                "type": "string",
                "enum": ["HOLD", "SIGNAL_FLEXIBILITY", "CONCEDE"],
                "description": (
                    "HOLD: maintain your position, no concessions. "
                    "SIGNAL_FLEXIBILITY: indicate willingness to negotiate, without full concession. "
                    "CONCEDE: accept a deal substantially on the opponent's terms."
                ),
            },
            "concession_probability": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": (
                    "Your honest probability estimate that you will concede "
                    "within the next 5 periods."
                ),
            },
            "delay_cost_implied": {
                "type": "number",
                "minimum": 0.0,
                "description": (
                    "On a scale of 0-10, how painful is continued delay for your side "
                    "right now? 0=painless, 10=catastrophic."
                ),
            },
            "belief_opponent_delay_cost": {
                "type": "number",
                "minimum": 0.0,
                "description": (
                    "Your estimate of opponent's delay cost on the same 0-10 scale. "
                    "Higher = they are feeling more pain."
                ),
            },
            "public_statement": {
                "type": "string",
                "description": (
                    "The public statement your faction releases this period "
                    "(1-2 sentences, consistent with your action)."
                ),
                "maxLength": 250,
            },
        },
        "required": [
            "reasoning",
            "action",
            "concession_probability",
            "delay_cost_implied",
            "belief_opponent_delay_cost",
            "public_statement",
        ],
    },
}

EXTRACT_COSTS_TOOL: dict = {
    "name": "extract_costs",
    "description": "Extract implied delay costs from a negotiation transcript.",
    "input_schema": {
        "type": "object",
        "properties": {
            "agent_a_cost_estimate": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 10.0,
            },
            "agent_b_cost_estimate": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 10.0,
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "reasoning": {
                "type": "string",
                "maxLength": 400,
            },
        },
        "required": [
            "agent_a_cost_estimate",
            "agent_b_cost_estimate",
            "confidence",
            "reasoning",
        ],
    },
}
