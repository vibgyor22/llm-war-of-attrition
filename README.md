# LLMs in a War of Attrition: Fiscal Brinkmanship and the U.S. Debt Ceiling

This project asks a political-economy question:

> If two large language model agents are placed inside a debt-ceiling crisis, do they behave like players in a war of attrition?

In a war of attrition, both sides prefer agreement to disaster, but each side wants the other side to accept the painful concession. The result is delay. Each period of delay hurts both sides, but not equally. The side for whom waiting becomes more costly should concede first.

The theoretical backbone is Alesina and Drazen (1991), "Why are stabilizations delayed?" Their model explains why fiscal stabilizations are often postponed even when everyone knows adjustment is needed: groups disagree over who pays, so they wait for the other side to fold.

This repo turns that idea into an LLM-agent experiment.

Two agents negotiate over a U.S. debt-ceiling deadline:

- **HAWK / Republican fiscal conservative**: wants spending cuts, opposes tax increases, cares about toughness and fiscal credibility.
- **DOVE / Democratic fiscal liberal**: wants to protect social programs, prefers revenue offsets, wants to avoid normalizing debt-ceiling hostage-taking.

Each side sees the same crisis state each period: market stress, VIX, Treasury bill rates, polling, debt/GDP, deficit projections, election timing, the X-date countdown, and the opponent's last action. They privately reason, choose an action, and report how costly further delay feels.

The outcome is not scripted. The simulation records who holds out, who signals flexibility, who concedes, and when.

---

## Live Dashboard

This repository is designed around a Streamlit dashboard:

```text
dashboard/app.py
```

If deployed on Streamlit Community Cloud, use:

```text
Repository: vibgyor22/llm-war-of-attrition
Branch: main
Main file path: dashboard/app.py
```

The dashboard is pure visualization. It reads the committed results under `outputs/` and does not need API keys at runtime.

---

## The Core Idea

Debt-ceiling bargaining has the structure of a war of attrition:

1. Both sides want to avoid default.
2. Each side wants the other side to absorb the political cost.
3. Waiting creates real costs: market volatility, rising bill yields, lower approval, deadline pressure, and reputational risk.
4. Each side tries to infer whether the opponent is under more pain.
5. The side with the higher effective delay cost should concede first.

In the code, each negotiation lasts up to 30 periods, where each period is one day in the countdown to the X-date.

At each period:

```text
shared crisis data -> HAWK private reasoning -> HAWK action
                   -> DOVE private reasoning -> DOVE action
                   -> transcript record
                   -> stop if either side concedes
```

The possible actions are:

```text
HOLD
SIGNAL_FLEXIBILITY
CONCEDE
```

If HAWK concedes, DOVE wins. If DOVE concedes, HAWK wins. If no one concedes in 30 periods, the run is censored.

---

## What The Agents See

The agents receive a structured state message each period. The state contains:

| Variable | Meaning |
|---|---|
| `period` | Current negotiation day, 0 to 29 |
| `date` | Calendar date in the crisis window |
| `days_to_xdate` | Days remaining until the debt-ceiling deadline |
| `vix` | CBOE VIX volatility index |
| `tbill_4wk` | 4-week Treasury bill rate |
| `market_stress_index` | Composite stress measure built from VIX and T-bill spread |
| `polling_approval_pct` | President approval, interpolated to daily frequency |
| `debt_gdp_ratio` | Federal debt as a percent of GDP |
| `deficit_projection_bn` | Deficit projection in billions |
| `election_days_out` | Days until the next election |
| `opponent_last_action` | What the other agent did in the previous period |
| `narrative_injection` | Condition-specific crisis language |

Most numeric data is symmetric: both agents see the same crisis state. The asymmetry comes from their role prompts. The same VIX shock can mean different things to each agent:

- HAWK may read rising yields as damage to fiscal credibility and investor confidence.
- DOVE may read rising market stress as pressure on HAWK's donor and business coalition.

That interpretive asymmetry is the experiment.

---

## Where The Data Comes From

### FRED Market and Macro Data

`src/data/fred_fetcher.py` downloads these series from FRED:

| Series | Used For |
|---|---|
| `VIXCLS` | VIX volatility |
| `TB4WK` | 4-week Treasury bill rate |
| `GFDEGDQ188S` | Federal debt as percent of GDP |
| `MTSDS133FMS` | Federal deficit/surplus |

The raw downloaded files live under `data/raw/` when the data pipeline is run. Raw data is ignored by Git.

### Polling Data

Polling files are stored under:

```text
data/raw/polling_2011.csv
data/raw/polling_2013.csv
data/raw/polling_2023.csv
```

The simulation mainly uses president approval:

- `obama_approval_pct` for 2011 and 2013
- `biden_approval_pct` for 2023
- synthetic president approval for the fictional counterfactual

Polling is sparse, so `src/data/episode_builder.py` interpolates it into daily values for each 30-day episode.

### Processed Episode Data

The final per-period episode files are stored in:

```text
data/processed/
```

Each episode parquet file contains 30 rows, one per simulated period.

---

## Market Stress Calculation

`src/data/preprocessor.py` constructs a daily `market_stress_index`.

First it aligns VIX, T-bill, debt/GDP, and deficit data to daily frequency.

Then it computes:

```text
tbill_spread = tbill_4wk - rolling_30_day_mean(tbill_4wk)
```

Then it normalizes VIX and T-bill spread to the 0-1 range:

```text
vix_norm = minmax(VIX)
tbill_spread_norm = minmax(tbill_spread)
```

Then it combines them:

```text
market_stress_index = 0.5 * vix_norm + 0.5 * tbill_spread_norm
```

Interpretation:

- Near 0: calm market environment
- Near 1: severe stress or panic

This index is not a claim about true default probability. It is an experimental proxy for financial pressure.

---

## Episodes

The project uses three historical episodes and one fictional counterfactual:

| Episode | Window | Description |
|---|---|---|
| 2011 | Budget Control Act standoff | Real debt-ceiling crisis ending near August 2, 2011 |
| 2013 | Shutdown and debt limit | Real government shutdown/debt-ceiling episode ending near October 17, 2013 |
| 2023 | Fiscal Responsibility Act | Real debt-ceiling episode ending near June 2023 |
| 2025 counterfactual | Fictional | Synthetic out-of-sample episode designed to reduce memorization concerns |

Each episode has an `xdate`, a `t0_date`, fiscal context, polling context, and election timing in `configs/episodes/`.

---

## Stress Conditions A-E

Conditions A-E are controlled counterfactual stress environments. They are not separate historical events. They scale the same base episode into different crisis intensities.

| Condition | VIX Multiplier | T-bill Spike | Stress Floor | Approval Offset | Interpretation |
|---|---:|---:|---:|---:|---|
| A | 0.7 | 0 bps | 0.10 | +3 | Calm |
| B | 1.0 | +15 bps | 0.35 | 0 | Baseline |
| C | 1.4 | +40 bps | 0.65 | -8 | High stress |
| D | 1.8 | +80 bps | 0.82 | -15 | Near crisis |
| E | 2.5 | +150 bps | 0.95 | -25 | Extreme crisis |

The purpose is identification. Using multipliers lets us ask:

> Holding the episode fixed, what happens if market stress is calmer or more severe?

The condition transformation happens in `src/simulation/game_engine.py`.

---

## Delay Cost

Delay cost means:

> How painful is continued waiting for this side right now?

Examples of pressure that can raise delay cost:

- VIX rising
- T-bill rates spiking
- the X-date getting closer
- approval falling
- fear of default
- fear of political blame
- reputational damage
- internal caucus pressure
- the opponent refusing to move

The live simulation does **not** calculate delay cost with a fixed formula.

Instead, each LLM agent reports it as part of its structured decision:

```json
{
  "action": "SIGNAL_FLEXIBILITY",
  "concession_probability": 0.42,
  "delay_cost_implied": 7.0,
  "belief_opponent_delay_cost": 6.5,
  "public_statement": "..."
}
```

That means delay cost is a model-produced strategic judgment, not a deterministic transformation of VIX or approval.

The analysis then constructs three delay-cost measures:

| Measure | Definition |
|---|---|
| `S` self-reported | Mean of `delay_cost_implied` across active periods |
| `B` behavioral | `1 / first period of SIGNAL_FLEXIBILITY or CONCEDE` |
| `J` judge-extracted | Separate judge LLM estimates delay cost from anonymized transcripts |

The main cost ratio is:

```text
cost_ratio_S = hawk_S / dove_S
```

If this ratio is greater than 1, HAWK reported higher delay cost than DOVE.

---

## What Is Logged

Each simulation produces a JSONL transcript in:

```text
outputs/transcripts/
```

Line 1 is metadata:

```json
{
  "sim_id": "2011_A_sim000",
  "episode_id": "2011",
  "condition_id": "A",
  "winner": "DOVE",
  "concession_period": 27,
  "conceding_agent": "HAWK"
}
```

Each later line is one period record containing:

- state variables
- HAWK action
- HAWK reasoning and delay cost
- DOVE action
- DOVE reasoning and delay cost
- Bayesian belief trackers
- whether the game continues

Compiled results are stored in:

```text
outputs/results/simulation_results.parquet
outputs/results/period_level_data.parquet
```

These files are committed so the dashboard can run without API calls.

---

## What The Current Outputs Show

The committed dashboard is based on 80 negotiation runs:

```text
4 episodes x 5 stress conditions x 2 replicates x 2 mask states = 80 runs
```

Broad patterns in the saved transcripts:

- Higher stress pushes both agents from `HOLD` toward `SIGNAL_FLEXIBILITY`.
- Higher VIX is associated with higher self-reported delay cost.
- Lower approval raises perceived political pressure.
- Near the X-date, concession probabilities rise sharply.
- Opponent behavior matters: when one side signals flexibility, the other usually softens next period.
- In the saved run set, HAWK concedes much more often than DOVE. This is an emergent output, not a scripted rule.

One important implementation detail: if both agents concede in the same period, the game engine currently treats HAWK's concession as authoritative, so DOVE is recorded as winner. That affects outcome counts and should be remembered when interpreting "who conceded first."

---

## Theory and Empirical Tests

The project compares LLM outcomes to the Alesina-Drazen logic:

```text
Higher delay cost -> higher probability of conceding first
Higher market stress -> higher concession hazard
Closer deadline -> higher pressure to resolve
```

Analysis modules:

| File | Purpose |
|---|---|
| `src/theory/alesina_drazen.py` | Analytical war-of-attrition baseline |
| `src/theory/bayesian_belief.py` | Researcher-side belief updates over delay costs |
| `src/analysis/cost_measures.py` | Self-reported, behavioral, and judge delay-cost measures |
| `src/analysis/regression.py` | OLS, logit, and Cox hazard models |
| `src/analysis/theory_benchmark.py` | Compare empirical concession timing with AD-style theory |
| `src/analysis/judge_evaluator.py` | Judge LLM estimates delay costs from anonymized transcripts |

The headline regression logic is:

```text
concession_time ~ cost_ratio + market_stress + election_pressure + episode
```

and:

```text
hazard ~ cost_ratio + market_stress + election_pressure
```

---

## Repository Structure

```text
.
|-- configs/
|   |-- conditions/          # A-E stress treatments
|   `-- episodes/            # episode metadata and fiscal context
|-- dashboard/
|   |-- app.py               # Streamlit dashboard
|   `-- components/
|-- data/
|   |-- processed/           # committed parquet datasets used by dashboard
|   |-- raw/                 # ignored raw FRED/polling inputs
|   `-- schemas/
|-- outputs/
|   |-- results/             # committed compiled results
|   `-- transcripts/         # committed simulation transcripts
|-- paper/
|   `-- paper_outline.md
|-- src/
|   |-- agents/              # HAWK, DOVE, base LLM wrapper, tool schemas
|   |-- analysis/            # cost measures, regressions, judge evaluation
|   |-- cache/               # SQLite LLM response cache
|   |-- data/                # FRED fetch, preprocessing, episode build
|   |-- simulation/          # game engine, condition runner, batch runner
|   |-- theory/              # Alesina-Drazen model and Bayesian belief tracker
|   `-- viz/                 # Plotly figure helpers
|-- tests/
|-- README.md
|-- requirements.txt
|-- pyproject.toml
`-- run_sim.py
```

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/vibgyor22/llm-war-of-attrition.git
cd llm-war-of-attrition
```

### 2. Install

```bash
python -m pip install -r requirements.txt
```

### 3. Run the dashboard

```bash
streamlit run dashboard/app.py
```

This uses the committed `outputs/` files and should not require API keys.

### 4. Optional: configure API keys for new simulations

Copy the example environment file:

```bash
cp .env.example .env
```

Then fill in:

```text
ANTHROPIC_API_KEY=...
FRED_API_KEY=...
```

Use `LLM_CACHE_MODE=read-write` for new LLM calls and `LLM_CACHE_MODE=read-only` for reproduction from cache.

---

## Running The Pipeline

The Makefile provides common commands:

| Target | What It Does |
|---|---|
| `make setup` | Install dependencies |
| `make data` | Fetch FRED data, preprocess, and build episode files |
| `make theory` | Run the Alesina-Drazen theory demo |
| `make run` | Smoke simulation run |
| `make run-full` | Larger simulation run |
| `make dashboard` | Launch Streamlit |
| `make test` | Run tests |

Note: if running modules directly, the simulation package path is `src.simulation`, not `src.simulations`.

---

## Reproducibility and Safety

- `.env` is ignored and should never be committed.
- `data/raw/` is ignored because raw downloads are regenerable.
- `outputs/results/` and `outputs/transcripts/` are committed so the dashboard is self-contained.
- LLM calls are cached with SQLite to reduce cost and improve reproducibility.
- Historical episodes may be in model training data, so the design includes masked runs and a fictional counterfactual episode.

---

## Limitations

This is an experimental simulation, not a causal estimate of real congressional behavior.

Important limitations:

- Delay cost is mostly LLM-reported, not directly observed.
- HAWK and DOVE receive mostly symmetric numeric inputs.
- Some side-specific YAML fields, such as ideology scores, are currently not deeply wired into the prompt logic.
- A-E stress regimes are counterfactual treatments, not actual historical paths.
- The model can still carry historical priors despite masking.
- Simultaneous concession handling affects who is counted as conceding first.

These limitations are part of the research question: the project is testing whether LLM agents can reproduce theory-like strategic behavior under controlled conditions, not claiming they perfectly model real politics.

---

## Citation

If you use this code, dashboard, or results, please cite:

```bibtex
@misc{attrition2026,
  author = {Vibhor Vanvani},
  title = {LLMs in a War of Attrition: Fiscal Brinkmanship and the U.S. Debt Ceiling},
  year = {2026},
  note = {Working paper and simulation dashboard},
  url = {https://github.com/vibgyor22/llm-war-of-attrition}
}
```

Reference:

```text
Alesina, Alberto, and Allan Drazen. 1991.
"Why Are Stabilizations Delayed?"
American Economic Review 81(5): 1170-1188.
```

---

## License

MIT License. See `LICENSE`.
