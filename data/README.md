# Data inputs

These files were extracted from the team-provided uploads so the project can run without the original Excel workbook.

| File | Direct source | Purpose |
|---|---|---|
| `gaia_quarterly_panel.csv` | `GAIA_AI_信号台_2026-10-29(1).xlsx`, sheet `05_季度指标面板` | 2023Q3–2026Q3 quarterly stock features and return labels |
| `account_snapshot.csv` | Same workbook, sheet `01_日报` | Account snapshot dated 2026-10-29 |
| `positions_snapshot.csv` | Same workbook, sheet `03_持仓快照` | SPYM and 12-stock position snapshot |
| `base_weights.csv` | `allocation_12_plus_etf(1).xlsx` | Strategic 60% SPYM / 40% stock-sleeve weights |

The original quarterly workbook states that its upstream market data came from Sina Finance adjusted US daily prices and its fundamental data from SEC EDGAR XBRL. Those upstream records were not independently re-downloaded for this project.

Do not put StockTrak/WInS usernames, passwords, session cookies, or other credentials in this folder.
