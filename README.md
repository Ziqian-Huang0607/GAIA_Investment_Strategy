# GAIA 2.0 — Full Research System

**Repository:** https://github.com/Ziqian-Huang0607/GAIA_Investment_Strategy  
**Team:** Commonwealth Funding  
**Model:** GAIA

> **Purpose:** **GAIA** is **Commonwealth Funding’s** research model for the **2026–2027 Wharton Global High School Investment Competition** client case. It connects client liabilities, quantitative equity research, evidence-gated AI analysis, portfolio risk, and long-term funding analysis in one reproducible workflow.

The official competition asks teams to build a long-term strategy for a client case and evaluates strategy quality, client alignment, research, analysis, and communication—not portfolio performance alone. See the [official competition overview](https://globalyouth.wharton.upenn.edu/competitions/investment-competition/) and [General Rules & Roles](https://globalyouth.wharton.upenn.edu/competitions/investment-competition/rules-roles/).

This is an independent student project, **not an official Wharton tool and not affiliated with or endorsed by Wharton**. It supports research for the IPS, trading-notes documentation, and final report; it is not investment advice or an automated trading system.

A real, installable Python project implementing Commonwealth Funding's complete research workflow:

**Client → Liability → Macro Regime → Sector Opportunity → Quant ML → Human Deep Dive → Narrative Agents → Uncertainty Gate → GFSS/GCC → Dynamic Risk Budget → Allocation Audit → Funding Probability → 2031 Commitment → 2033 Reserve**

It includes source modules, editable data, configuration, a CLI, automated tests, CSV/JSON outputs, and a full Excel audit report. It does **not** log into StockTrak/WInS, place orders, or replace the investment committee.

> **Competition AI compliance:** the official rules state that AI-generated work may not be submitted as your own and that included AI-generated material must be properly cited. Independently verify this system’s outputs, retain source and demo labels, and follow the official rules when writing any submission.

For GitHub repository text and topics, see [`GITHUB_REPOSITORY.md`](GITHUB_REPOSITORY.md). For the sources used by the project, see [`REFERENCES.md`](REFERENCES.md).

## Commonwealth Funding

Developed by **Ziqian Huang, Ethan Qin, Ka Ngat Li, Eric Liu, and YuTong Lin**.

The repository includes a GitHub-compatible [`CITATION.cff`](CITATION.cff), so visitors can use GitHub's **Cite this repository** button to cite this software. The separate [`REFERENCES.md`](REFERENCES.md) file cites the competition, client-case inputs, upstream data documentation, and methods used by the project.

## 1. Quickstart

Requires Python 3.10 or newer.

Clone the repository:

```bash
git clone https://github.com/Ziqian-Huang0607/GAIA_Investment_Strategy.git
cd GAIA_Investment_Strategy
```

### Windows

```bat
cd GAIA_Investment_Strategy
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .[dev]
gaia-ai full --as-of 2026-10-29
pytest
```

Or double-click `run.bat` after Python is installed. To reproduce the included demo date:

```bat
run.bat --as-of 2026-10-29
```

### macOS / Linux

```bash
cd GAIA_Investment_Strategy
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
gaia-ai full --as-of 2026-10-29
pytest
```

Or:

```bash
bash run.sh --as-of 2026-10-29
```

### Run without installing the package

From the project root:

```bash
PYTHONPATH=src python -m gaia_ai full --as-of 2026-10-29
PYTHONPATH=src python -m pytest
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m gaia_ai full --as-of 2026-10-29
python -m pytest
```

For a fast integration check, override the default 3,000 paths per scenario:

```bash
gaia-ai full --as-of 2026-10-29 --simulation-paths 200 --output-dir outputs/smoke
```

## 2. Commands

| Command | Purpose |
|---|---|
| `gaia-ai full` | Run all GAIA modules and generate the complete audit report |
| `gaia-ai run` | Run the original quant training → backtest → prediction → daily-signal subset |
| `gaia-ai train` | Train, walk-forward backtest, and score the latest quarter |
| `gaia-ai daily` | Generate daily position and trade-review outputs from existing predictions |

Examples:

```bash
gaia-ai full --as-of 2026-10-29 --config config/default.json --output-dir outputs/full_01
gaia-ai run --as-of 2026-10-29
gaia-ai train --data data/gaia_quarterly_panel.csv --as-of 2026-10-29
gaia-ai daily --predictions outputs/latest_predictions.csv
```

If `--as-of` is omitted, the current date is used. Use the explicit demo date for exact reproduction.

## 3. Implemented GAIA modules

### 3.1 Client goals and liability mapping

`data/client_profile.json` encodes the case facts:

- $300,000 contribution at the beginning of 2027.
- $150,000 contribution at the beginning of 2028.
- Ten $50,000 payments at the beginning of each year from 2033 through 2042.
- Co-sponsor communication in 2031.
- Reserve established before the first 2033 payment or facility contribution.
- No sponsor or other outside funding is included in Operating Funding Confidence.

`liability.py` calculates the payment schedule, beginning-of-year present values, funding ratios, liability phases, reserve glidepath, and commitment percentiles.

### 3.2 Macro regime and Sector Opportunity Score

`sector.py` classifies Recovery, Overheat, Stagflation, Recession, or Disinflationary Slowdown from editable growth/inflation directions.

The Sector Opportunity Score combines:

- 80% transparent scorecard: growth, earnings revisions, profitability, valuation, and risk-adjusted momentum.
- 20% regime fit.

The supplied scorecard covers only four candidate sectors, not the full GICS universe. Missing earnings-revision breadth receives a neutral 0.5 component and is explicitly flagged.

### 3.3 Quantitative ML funnel

The original 3B/3C funnel remains intact. The supervised model predicts **next-quarter return relative to the S&P 500**, not a precise daily price.

Features:

1. `3B综合分`
2. `3C成长`
3. `3C质量`
4. `E/P = 1 / P/E(TTM)`

Preprocessing uses quarterly cross-sectional z-scores with `ddof=0`. Fundamental records older than 400 days have growth, quality, and E/P invalidated. Labels are eligible only after the following quarter has fully ended. Missing values remain missing.

Estimator:

```python
Ridge(alpha=10, positive=True, fit_intercept=True)
```

Validation is expanding-window walk-forward: each test quarter is fitted only on earlier quarters.

Reproduced with `--as-of 2026-10-29`:

| Metric | Result |
|---|---:|
| Out-of-sample quarters | 9 |
| Final training rows | 595 |
| IC mean | 0.061855 |
| ICIR | 0.231774 |
| Direction hit rate | 55.45% |
| Top5 − Bottom5 average quarterly spread | 3.01 percentage points |
| Out-of-sample RMSE | 12.98 percentage points |

These are historical research statistics, not expected future returns.

### 3.4 Human deep dive

`data/human_research.csv` contains business, moat, management, catalyst, bear-risk, fundamental, confidence, rationale, and source fields.

Every supplied row has `human_approved=False`, because the uploaded 3D workbook identifies its research as AI-simulated. Only a real reviewer may change this field after checking the underlying sources. Until then, candidates can produce research scores but receive zero approved stock target weight.

### 3.5 Narrative intelligence and evidence traces

Eight dimensions are scored from 1–5 with confidence, direction of change, and evidence:

- Demand Strength
- Pricing Power
- Execution
- Capex Pressure
- Balance-Sheet Risk
- Governance
- Competitive Position
- Disruption Risk

Every evidence row retains ticker, document ID/date, source URL, agent, dimension, score, confidence, direction, evidence text, provider, and demo status.

The default provider is **deterministic extractive fallback, not an LLM**. It always triggers review. Demo documents also trigger review.

### 3.6 Five-agent committee and AI Disagreement Score

The five roles are:

1. Fundamental Analyst
2. Risk Analyst
3. Macro Analyst
4. Devil's Advocate
5. Client-Liability Analyst

AI Disagreement Score (ADS) is the population standard deviation of agent overall scores on the 1–5 scale. It measures disagreement; it is not a probability of loss.

### 3.7 AI Uncertainty Gate

A candidate can be blocked by:

- offline extractive provider rather than a real LLM;
- low AI confidence;
- high agent disagreement;
- demo documents;
- missing human deep-dive approval.

Passing an AI gate still means **progress to human investment committee**, never automatic trading.

### 3.8 GFSS, confidence adjustment, GoalFit, and GCC

GAIA Final Stock Score combines transparent quant, fundamentals, narrative delta, catalysts, management, AI consensus, and risk. The adjusted score is multiplied by a confidence measure reflecting data, research, and AI confidence.

Goal-Conditioned Conviction is:

```text
GCC = Adjusted GFSS × GoalFit − RiskPenalty
```

GoalFit and RiskPenalty become more conservative as the liability phase moves from Growth to Transition, Protection, and Liability Segmentation.

`gfss_weight_sensitivity()` shocks each GFSS weight by ±20%, renormalizes positive weights, and reports rank correlation with the base result. This is a robustness check, not return-based weight tuning.

### 3.9 Dynamic Risk Capacity and allocation audit

`risk.py` combines time, funding progress, market volatility, drawdown, and liability targets to produce Growth, Stability, and FutureLiability weights.

Being underfunded does **not** authorize extra risk. Being ahead of the funding path harvests risk.

The allocation report distinguishes:

- `research_weight`: a fully allocated research illustration;
- `approved_target_weight`: zero for gated candidates;
- `action`: `RESEARCH_ONLY__NO_TRADE` for unapproved candidates.

Audits cover stock-sleeve totals, single-stock caps, sector redundancy, strategic SPYM weight, human approval, and bucket totals. Same-sector concentration is explicitly labelled as a redundancy proxy, not an estimated daily-return correlation.

The original daily module still supports the 60% SPYM / 40% stock sleeve, ±35% tilt, 2%–6% individual-stock bounds, 0.8-percentage-point review threshold, whole shares, $25 commissions, cash-constrained buys, and the 200-trade cap.

### 3.10 Funding Probability Engine, 2031 commitment, and 2033 reserve

The simulator uses correlated Student-t returns with six degrees of freedom, annual rebalancing, beginning-of-year cash flows, and a fixed random seed. The full command applies the dynamic risk-budget glidepath through 2042; the schedule uses the risk module's transparent illustrative funding-path roll-forward, not a market forecast.

Default internal research assumptions:

| Bucket | Mean annual return | Volatility |
|---|---:|---:|
| Growth | 7% | 16% |
| Stability | 3% | 5% |
| FutureLiability | 2% | 1% |
| Reserve | 2% | 1% |

These are **not official WGYS parameters or market forecasts**. Edit them in `config/default.json` and rerun the scenarios.

Operating Funding Confidence (OFC) is the fraction of simulated paths in which all ten payments are funded **without sponsor support**. The report includes 90%, 95%, 97.5%, and 99% confidence-threshold comparisons and payment-year funding probabilities.

Scenarios:

- base
- bull
- bear
- recession
- equity crash
- rate shock
- sequence-of-returns stress

The 2031 commitment report shows the distribution of the contingent 2033 reserve shortfall, including P5/P10/P25/P50/P75/P90/P95/P97.5/P99, expected contribution, expected positive contribution, maximum contribution, probability of zero contribution, and the configured P10–P50 range. Zero is a valid result; the facility is not treated as base-case funding.

The 2033 reserve target is:

```text
PV of remaining beginning-of-year payments + $50,000 flexibility buffer
```

With the supplied 2% reserve discount rate, the target is $508,111.84 before the first 2033 payment. The responsible facility contribution is the simulated shortfall after available portfolio assets are transferred to the reserve.

## 4. Optional real-LLM provider

The project includes an OpenAI-compatible `/chat/completions` adapter. No credentials are hardcoded.

```bash
export GAIA_LLM_BASE_URL="https://your-endpoint.example/v1"
export GAIA_LLM_API_KEY="your-key"
export GAIA_LLM_MODEL="your-model"
export GAIA_LLM_TIMEOUT="90"  # optional

gaia-ai full --as-of 2026-10-29 --narrative-provider openai
```

Windows PowerShell:

```powershell
$env:GAIA_LLM_BASE_URL = "https://your-endpoint.example/v1"
$env:GAIA_LLM_API_KEY = "your-key"
$env:GAIA_LLM_MODEL = "your-model"
gaia-ai full --as-of 2026-10-29 --narrative-provider openai
```

The adapter requires strict JSON, all eight dimensions, valid score/confidence ranges, and exact document quotes. Missing credentials, malformed responses, omitted dimensions, or invented evidence raise errors rather than silently falling back to a pretend LLM result. Demo and human-approval gates still apply.

## 5. Project structure

```text
gaia-ai/
├── pyproject.toml
├── README.md                    # Project purpose and operating guide
├── REFERENCES.md                # Sources used by the project
├── GITHUB_REPOSITORY.md         # GitHub About text, topics, and release checklist
├── CITATION.cff                 # Metadata for citing this software
├── config/default.json
├── data/
│   ├── gaia_quarterly_panel.csv
│   ├── account_snapshot.csv
│   ├── positions_snapshot.csv
│   ├── base_weights.csv
│   ├── client_profile.json
│   ├── market_regime.csv
│   ├── sector_scorecard.csv
│   ├── human_research.csv
│   └── narrative_documents_demo.csv
├── src/gaia_ai/
│   ├── cli.py                 # full / train / daily / run
│   ├── pipeline.py            # Full-system orchestrator
│   ├── data.py                # Point-in-time panel preparation
│   ├── model.py               # Ridge estimator
│   ├── backtest.py            # Walk-forward validation
│   ├── sector.py              # Regime and sector scores
│   ├── narrative.py           # Evidence, five agents, ADS, AI gate
│   ├── conviction.py          # GFSS, confidence, GoalFit, GCC
│   ├── liability.py           # Client cash flows and commitments
│   ├── risk.py                # Dynamic risk capacity
│   ├── allocation.py          # Research/approved weights and audits
│   ├── simulation.py          # Funding Monte Carlo and scenarios
│   ├── portfolio.py           # Original daily whole-share workflow
│   └── full_reporting.py      # Full Excel audit report
├── tests/                     # Unit and end-to-end tests
├── outputs/                   # Generated CSV, JSON, XLSX
├── run.bat
└── run.sh
```

## 6. Main full-system outputs

| File | Purpose |
|---|---|
| `outputs/full_system_summary.json` | System status and key results |
| `outputs/GAIA_full_system_report.xlsx` | Multi-sheet human-readable audit report |
| `outputs/conviction.csv` | GFSS, confidence, GoalFit, GCC, and gates |
| `outputs/narrative_agents.csv` | Five-agent assessments |
| `outputs/narrative_evidence.csv` | Dimension-level evidence traces |
| `outputs/allocation_research.csv` | Research and approved weights kept separate |
| `outputs/allocation_audit.csv` | Constraint and governance checks |
| `outputs/funding_scenarios.csv` | OFC and scenario summaries |
| `outputs/ofc_thresholds.csv` | Confidence-threshold comparisons |
| `outputs/payment_probability.csv` | Funding probability by payment year |
| `outputs/commitments_2031.csv` | Facility-commitment distributions |
| `outputs/annual_paths.csv` | Annual P10/P50/P90 asset paths |
| `outputs/reserve_glidepath.csv` | Reserve and liability-bucket schedule |
| `outputs/gfss_sensitivity.csv` | GFSS weight robustness checks |
| `outputs/facility_samples_base.csv` | Reproducible base-scenario path samples |

CSV and JSON are the machine-readable outputs. Excel is a reporting convenience, not the model itself.

## 7. Daily manual-input workflow

Update `data/account_snapshot.csv` with the five StockTrak/WInS values and trades used. Update `data/positions_snapshot.csv` with ticker, whole-share quantity, and latest reference price. Then run:

```bash
gaia-ai daily
```

Review `outputs/trade_review.csv` and `outputs/daily_summary.json`. The program never requests account credentials and never submits orders.

## 8. Tests

```bash
pytest
```

Current result: **17 passed**.

The suite covers:

- incomplete-label masking, stale fundamentals, z-scores, and unique panel keys;
- walk-forward counts and exact reproduced metrics;
- daily portfolio bounds, whole shares, cash constraints, and trade limits;
- exact ten-payment schedule and beginning-of-year present value;
- commitment percentiles and zero-contribution handling;
- five agents, 40 evidence rows per document, offline/demo gates, and exact-quote enforcement;
- regime classification and neutral missing-revision handling;
- risk harvesting only when ahead of the funding path;
- Monte Carlo reproducibility, reserve target, sponsor exclusion, and OFC thresholds;
- end-to-end full-system status, evidence counts, allocation audits, and report creation.

## 9. Citation

Suggested plain-text citation:

> Huang, Z., Qin, E., Li, K. N., Liu, E., & Lin, Y. (2026). *GAIA: Liability-Driven Investment Research System* (Version 2.0.0) [Computer software]. Commonwealth Funding. https://github.com/Ziqian-Huang0607/GAIA_Investment_Strategy

BibTeX:

```bibtex
@software{commonwealth_funding_gaia_2026,
  author  = {Huang, Ziqian and Qin, Ethan and Li, Ka Ngat and Liu, Eric and Lin, YuTong},
  title   = {GAIA: Liability-Driven Investment Research System},
  year    = {2026},
  version = {2.0.0},
  note    = {Commonwealth Funding},
  url     = {https://github.com/Ziqian-Huang0607/GAIA_Investment_Strategy}
}
```

The repository URL is included in `CITATION.cff`, the package metadata, and the citation formats above.

## 10. Limitations and governance

- The default run is an **offline demo**, not a real-LLM research run.
- Uploaded 3D research is AI-simulated and remains human-unapproved.
- Only nine out-of-sample quarters are available; model confidence is limited.
- The universe is based on a 2026 candidate list, so survivorship bias remains.
- The underlying panel was supplied by the team; this project does not independently re-download every upstream source record.
- The previous daily 20-day momentum/volatility layer is not reproduced because its original code and daily time series were not supplied.
- Top5−Bottom5 is a cross-sectional research spread, not a transaction-cost-adjusted portfolio backtest.
- Sector rankings cover only supplied candidate sectors.
- Simulation results are assumption-dependent and are not guarantees or official Wharton parameters.
- The system provides decision support only. Human research approval and investment-committee review remain mandatory.
