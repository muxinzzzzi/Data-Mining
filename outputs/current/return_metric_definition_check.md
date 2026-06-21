# Return Metric Definition Check

This check uses the current local main-project outputs under `outputs/current/` and the main-project source files that define the backtest metrics. No model was retrained and no strategy was recalculated.

## Files Checked

- `outputs/current/summary.md`
- `outputs/current/final_strategy_comparison.csv`
- `outputs/current/report_tables/Table4_strategy_results.csv`
- `outputs/current/strategy_daily_all.csv`
- `outputs/current/report_figures/fig1_equity_curve.png` data source: `scripts/generate_final_report_assets.py` reads `outputs/current/strategy_daily_all.csv`
- Metric definition source:
  - `src/config.py`: `INITIAL_CAPITAL = 100_000.0`
  - `src/strategy/backtest.py`: daily equity construction
  - `src/strategy/metrics.py`: `total_return` and `excess_return_vs_buy_hold` formulas

## 1. What Does `total_return` Mean?

`final_strategy_comparison.csv` field `total_return` is cumulative return in decimal form, not final wealth multiple and not a percent-formatted number.

Code definition in `src/strategy/metrics.py`:

```python
total_return = float(equity.iloc[-1] / INITIAL_CAPITAL - 1)
```

Therefore:

```text
total_return = final wealth / initial capital - 1
final wealth multiple = 1 + total_return
cumulative return percent = total_return * 100%
```

The column name `total_return` is potentially misleading because a value such as `1.1087` may look like a wealth multiple. In this project output, it means cumulative return of `+110.87%`, and the final wealth multiple is `2.1087x`.

## 2. Buy-and-Hold Interpretation

`buy_hold total_return = 1.1087406833801303`.

Correct interpretation:

- Cumulative return: `+110.8741%`
- Final wealth multiple: `2.1087406833801303x`
- Not: final wealth multiple `1.1087`
- Not: cumulative return `+10.87%`

Evidence from `outputs/current/strategy_daily_all.csv`:

| item | value |
| --- | ---: |
| Initial capital | RMB 100,000.00 |
| First plotted trade date | 2025-01-03 |
| First available equity after first return | RMB 95,217.17 |
| Last trade date | 2026-05-06 |
| Final equity | RMB 210,874.07 |
| Final wealth multiple | 2.1087406834 |
| Cumulative return from equity | 1.1087406834 |

The first row of the daily equity series is already after the first test-period return. The initial RMB 100,000 starting point is implicit in the formula, not a separate first row in `strategy_daily_all.csv`.

## 3. `ml_risk_budget_enhancement` Interpretation

`ml_risk_budget_enhancement total_return = 1.1134946344539691`.

Correct interpretation:

- Cumulative return: `+111.3495%`
- Final wealth multiple: `2.113494634453969x`
- Not: final wealth multiple `1.1135`
- Not: cumulative return `+11.35%`

Evidence from `outputs/current/strategy_daily_all.csv`:

| item | value |
| --- | ---: |
| Initial capital | RMB 100,000.00 |
| First plotted trade date | 2025-01-03 |
| First available equity after first return | RMB 96,153.74 |
| Last trade date | 2026-05-06 |
| Final equity | RMB 211,349.46 |
| Final wealth multiple | 2.1134946345 |
| Cumulative return from equity | 1.1134946345 |

## 4. Final Wealth with RMB 100,000 Initial Capital

| Strategy | total_return | Cumulative return | Final wealth multiple | Final wealth |
| --- | ---: | ---: | ---: | ---: |
| buy_hold | 1.1087406834 | +110.8741% | 2.1087406834x | RMB 210,874.07 |
| ml_risk_budget_enhancement | 1.1134946345 | +111.3495% | 2.1134946345x | RMB 211,349.46 |

Final wealth difference:

```text
RMB 211,349.46 - RMB 210,874.07 = RMB 475.40
```

## 5. Strict Meaning of Excess Return `+0.0048`

`excess_return_vs_buy_hold = 0.0047539510738388024` for `ml_risk_budget_enhancement`.

Code definition in `src/strategy/metrics.py`:

```python
excess_return = total_return - benchmark_total
```

Strict interpretation:

- `+0.0047539511` in decimal cumulative-return units
- `+0.4754 percentage points` of cumulative return versus buy-and-hold
- Equivalent to `+0.4754% of initial capital`, i.e. about `RMB 475.40` on RMB 100,000
- Not `+4.8%`

Recommended wording: "The strategy outperforms buy-and-hold by 0.48 percentage points, equivalent to RMB 475 on a RMB 100,000 initial portfolio."

## 6. Current PDF / Report Error Check

No final LaTeX or final PDF report file was found under the main project, excluding `.venv` and `Data-Mining-ljn`. The only PDFs found in `outputs/current/` are report-figure PDFs, not a full course report.

Current output text check:

- `outputs/current/CURRENT_RESULT_README.md` writes `Buy-and-hold total return: 1.1087 (110.87%)`. This is correct if `total return` is interpreted as cumulative return in decimal form.
- `outputs/current/CURRENT_RESULT_README.md` writes best ML `total return: 1.1135 (111.35%)`. This is also correct.
- `outputs/current/summary.md` writes `Buy-and-hold total return: 1.1087` and `ML strategy total return: 1.1135`. These are numerically correct but wording is ambiguous because the values are not final wealth multiples.
- No checked current output text shows `RMB 110,870`, `RMB 111,350`, `+10.87%`, or `+11.35%` for the final results.

Conclusion:

- Writing `1.1087` as `+110.87%` is correct.
- Writing `1.1135` as `+111.35%` is correct.
- Writing final wealth as `RMB 210,870` / `RMB 211,350` is correct, allowing rounding.
- Writing final wealth as `RMB 110,870` / `RMB 111,350` would be incorrect.

## 7. Recommended Unified Report Wording

Use the following definitions consistently:

| Metric | Definition | Recommended display |
| --- | --- | --- |
| Final wealth multiple | `final wealth / initial capital` | `2.1087x`, `2.1135x` |
| Cumulative return | `final wealth / initial capital - 1` | `+110.87%`, `+111.35%` |
| `total_return` column | Same as cumulative return in decimal form | Avoid raw `1.1087` without explanation |
| Excess return | Strategy cumulative return minus buy-and-hold cumulative return | `+0.48 percentage points` |
| Final wealth | `initial capital * (1 + cumulative return)` | `RMB 210,874`, `RMB 211,349` |

Suggested final-report phrasing:

- "Buy-and-hold achieves a cumulative return of +110.87%, corresponding to a final wealth multiple of 2.1087x and final wealth of RMB 210,874 from RMB 100,000 initial capital."
- "The risk-budget enhancement strategy achieves a cumulative return of +111.35%, corresponding to a final wealth multiple of 2.1135x and final wealth of RMB 211,349."
- "The strategy's excess return is +0.48 percentage points, or about RMB 475 on the initial RMB 100,000 portfolio."

## 8. Global Replacement List for PDF / LaTeX Report

Apply these replacements if the expressions appear in the final PDF/LaTeX report:

| Incorrect or ambiguous expression | Replace with |
| --- | --- |
| `final wealth multiple = 1.1087` | `cumulative return = +110.87%; final wealth multiple = 2.1087x` |
| `final wealth multiple = 1.1135` | `cumulative return = +111.35%; final wealth multiple = 2.1135x` |
| `buy_hold return = +10.87%` | `buy_hold cumulative return = +110.87%` |
| `ml_risk_budget_enhancement return = +11.35%` | `ml_risk_budget_enhancement cumulative return = +111.35%` |
| `buy_hold final wealth = RMB 110,870` | `buy_hold final wealth = RMB 210,874` |
| `ml_risk_budget_enhancement final wealth = RMB 111,350` | `ml_risk_budget_enhancement final wealth = RMB 211,349` |
| `excess return = +4.8%` | `excess return = +0.48 percentage points` |
| Raw table heading `total_return` without definition | Add note: `total_return is cumulative return in decimal form, not final wealth multiple.` |

## Figure 1 Data Source Note

`fig1_equity_curve.png` was generated from `outputs/current/strategy_daily_all.csv` by `scripts/generate_final_report_assets.py`.

The plotting function reads each strategy's `equity` column and normalizes it by the first available plotted equity value:

```python
y = pd.to_numeric(frame["equity"], errors="coerce")
start = y.dropna().iloc[0]
ax.plot(frame["date"], y / start if start else y, ...)
```

Therefore Figure 1 is an equity growth visualization with `Start = 1.0` at the first plotted row, not a direct display of the final wealth multiple relative to the initial RMB 100,000. The final wealth and cumulative return should be taken from `strategy_daily_all.csv` and `final_strategy_comparison.csv` using the formulas above.

