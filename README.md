# LLM Agents, War of Attrition, and the US Debt Ceiling

## Abstract

This project tests whether large language model agents playing as Congressional and White House negotiators reproduce the strategic dynamics predicted by Alesina and Drazen (1991) during US federal debt-ceiling episodes. Claude-based agents (claude-haiku-4-5 for negotiating parties, claude-sonnet-4-6 as an impartial judge) are placed in historically-grounded fiscal negotiation environments drawn from four debt-ceiling crises. We measure concession timing, delay costs, and equilibrium type across five information conditions, and estimate survival-time and hazard models to test whether the war-of-attrition framework's core predictions — asymmetric delay costs, time-to-agreement distributions, and stabilization via unilateral concession — hold in the LLM setting.

---

## Quick Start

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd attrition

# 2. Install dependencies
make setup

# 3. Configure API keys
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY and FRED_API_KEY

# 4. Smoke run (2 sims/condition, ~$2 API cost)
make run

# 5. Launch the dashboard
make dashboard
```

---

## Project Structure

```
attrition/
├── src/
│   ├── data/
│   │   ├── fred_fetcher.py       # Pull macro series from FRED API
│   │   ├── preprocessor.py       # Clean and align time series
│   │   └── episode_builder.py    # Construct per-episode negotiation contexts
│   ├── theory/
│   │   └── alesina_drazen.py     # Analytical Alesina-Drazen (1991) baseline
│   ├── simulations/
│   │   ├── agents.py             # Claude agent wrappers (congress, executive, judge)
│   │   ├── environment.py        # Negotiation environment and state machine
│   │   └── batch_runner.py       # Condition × episode × replicate loop
│   ├── analysis/
│   │   └── main.py               # Econometric models, hypothesis tests
│   ├── viz/
│   │   └── figures.py            # Plotly figure generation
│   └── utils/
│       ├── cache.py              # SQLite-backed LLM response cache
│       └── cost_estimator.py     # Pre-run API cost projection
├── dashboard/
│   └── app.py                    # Streamlit interactive dashboard
├── tests/                        # pytest unit and integration tests
├── data/
│   ├── raw/                      # FRED downloads (gitignored)
│   └── processed/                # Cleaned datasets (gitignored)
├── outputs/
│   ├── transcripts/              # Full agent dialogue logs (gitignored)
│   ├── results/                  # Simulation summary tables (gitignored)
│   └── figures/                  # Publication figures (gitignored)
├── cache/
│   └── llm/
│       └── responses.sqlite      # LLM response cache (tracked in git)
├── pyproject.toml
├── requirements.txt
├── Makefile
└── .env.example
```

---

## Makefile Targets

| Target             | Description                                                    |
|--------------------|----------------------------------------------------------------|
| `make setup`       | Install all runtime dependencies from requirements.txt         |
| `make data`        | Fetch FRED series, preprocess, and build episode contexts      |
| `make theory`      | Compute Alesina-Drazen analytical baseline                     |
| `make cost-estimate` | Print projected API spend before running simulations         |
| `make run`         | Smoke run: 2 sims/condition, full pipeline (~$2)               |
| `make run-full`    | Full run: 25 sims/condition, full pipeline (~$50)              |
| `make reproduce-paper` | Reproduce paper results from cached responses (no API cost)|
| `make dashboard`   | Launch Streamlit results dashboard at localhost:8501           |
| `make test`        | Run pytest test suite with verbose output                      |
| `make clean`       | Delete outputs/ (cache/llm/ is preserved)                      |
| `make help`        | Print this target reference                                    |

---

## Research Design

### Episodes (4)
Historical US federal debt-ceiling crises used as negotiation contexts:
1. 2011 Budget Control Act standoff (August 2011)
2. 2013 government shutdown and debt limit (October 2013)
3. 2021 debt limit suspension expiry (October 2021)
4. 2023 Fiscal Responsibility Act (June 2023)

### Conditions (5)
Information environments that vary agent knowledge of opponent costs:
1. **Symmetric-full** — both agents observe each other's delay costs
2. **Asymmetric-private** — each agent knows only its own delay cost (baseline AD setting)
3. **Asymmetric-noisy** — private costs with noisy public signals
4. **Common-knowledge-cheap** — costs common knowledge but talk is cheap
5. **Commitment** — one agent can make binding offers

### Delay Cost Measures (3)
1. **Financial markets** — daily VIX change and credit-default-swap spread
2. **Real economy** — Treasury bill rate spike and consumer confidence drop
3. **Political** — approval-rating decline and intraparty cohesion index

### Econometric Models (4)
1. **Kaplan-Meier survival curves** — non-parametric time-to-agreement by condition
2. **Cox proportional hazards** — partial-likelihood estimate of concession hazard
3. **Accelerated failure time (Weibull)** — parametric delay duration model
4. **OLS / panel fixed effects** — concession timing regressed on cost asymmetry

---

## Citation

If you use this code or results, please cite:

```bibtex
@misc{attrition2026,
  author    = {[Author Name]},
  title     = {LLM Agents, War of Attrition, and the US Debt Ceiling},
  year      = {2026},
  note      = {Working paper},
  url       = {[repository URL]}
}
```

> Alesina, A., & Drazen, A. (1991). Why are stabilizations delayed? *American Economic Review*, 81(5), 1170–1188.
