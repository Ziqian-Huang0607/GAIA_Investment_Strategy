# GitHub Repository Setup

## Repository

**Name:** `GAIA_Investment_Strategy`  
**Owner:** `Ziqian-Huang0607`  
**URL:** https://github.com/Ziqian-Huang0607/GAIA_Investment_Strategy

## Copy-paste GitHub “About” description

> GAIA is Commonwealth Funding’s human-governed research model for the 2026–27 Wharton client case: quantitative ML, evidence-gated agents, risk budgeting, allocation audits, and liability funding simulations.

Short version:

> GAIA by Commonwealth Funding: quantitative ML, evidence-gated AI agents, risk budgeting, and client-liability funding analysis.

## Suggested GitHub topics

```text
python
finance
investment-research
quantitative-finance
machine-learning
ridge-regression
monte-carlo
risk-management
liability-driven-investing
student-research
```

## Purpose statement

This repository was built for **Commonwealth Funding’s research workflow in the 2026–2027 Wharton Global High School Investment Competition**. It supports analysis for the team’s client case, Investment Policy Statement, trading-notes documentation, and final strategy report.

The project is designed to make the research process auditable:

- map client contributions, operating payments, and reserve requirements;
- evaluate quantitative equity signals with point-in-time walk-forward validation;
- preserve narrative evidence, agent disagreement, confidence, and human-review gates;
- translate research conviction into risk-constrained research allocations;
- test funding adequacy under correlated fat-tailed simulations and stress scenarios;
- distinguish contingent facility commitments from base-case operating funding.

It is **not an official Wharton product, not affiliated with or endorsed by Wharton, not investment advice, and not an automated trading system**.

## Documentation map

- `README.md` — purpose, architecture, installation, commands, outputs, and limitations.
- `REFERENCES.md` — official competition links, team-provided materials, upstream data notes, and methodology/software references.
- `CITATION.cff` — metadata for citing this software repository through GitHub.
- `config/default.json` — editable model, risk, allocation, and simulation assumptions.
- `data/` — editable project inputs; demo and human-approval status are explicitly labelled.
- `outputs/` — generated locally; excluded from Git by default.

## Before making the repository public

1. **Check publication permission for team inputs.** The code references team-provided strategy and client documents, but those original documents are not bundled as public files. Do not upload official portal materials unless publication is permitted.
2. **Review personal information.** No team emails or birth dates are needed in the repository. Keep the uploaded roster screenshot out of Git.
3. **Keep credentials out.** Never commit WInS/StockTrak credentials, API keys, `.env` files, or account-login code. The project uses manual CSV inputs.
4. **Preserve demo labels.** Do not rename `narrative_documents_demo.csv` as real filings or transcripts, and do not change `human_approved` to `True` without genuine human review.
5. **Review competition AI rules.** AI-generated work may not be submitted as your own; included AI-generated material must be properly cited. Independently verify the analysis and write competition submissions in accordance with official rules.
6. **Choose a license deliberately.** Until a license is added, default copyright rules apply. Adding an open-source license is a separate team decision.
7. **Repository metadata is configured.** The URL is already included in `CITATION.cff`, `pyproject.toml`, the README, and the citation formats.

## Suggested first commit

```bash
git init
git add .
git commit -m "Initial GAIA investment research system"
git branch -M main
```

Connect the local repository to the existing GitHub repository and push:

```bash
git remote add origin https://github.com/Ziqian-Huang0607/GAIA_Investment_Strategy.git
git push -u origin main
```

If a remote named `origin` already exists, update it instead:

```bash
git remote set-url origin https://github.com/Ziqian-Huang0607/GAIA_Investment_Strategy.git
```

Do not commit generated `outputs/`; they are intentionally reproducible and ignored by Git.
