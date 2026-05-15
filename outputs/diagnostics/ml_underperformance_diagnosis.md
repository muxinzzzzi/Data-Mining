# ML Underperformance Diagnosis

## Headline
- Selected strategy: `ml_big_up_index_enhancement`
- Selected prediction source: `ExtraTreesDeep`, horizon `10D`
- Final-equity gap vs buy-and-hold: RMB -10943.48
- Final-equity gap as initial-capital percentage: -0.1094
- Arithmetic active-return sum: -0.0596

## Exposure Diagnostics
- Average position: 0.9399
- Minimum position: 0.8800
- Maximum position: 1.0000
- Days below full exposure: 257 (80.31%)
- Total turnover: 1.7000
- Average absolute position gap from 1.0: 0.0601
- Days with position <= 0.95: 134
- Days with position <= 0.90: 123
- Days with position <= 0.80: 0

## Active-Return Attribution
- Missed upside from reduced exposure on positive-return days: 0.1624
- Avoided downside from reduced exposure on negative-return days: 0.1045
- Turnover cost: 0.0017
- Net timing contribution: -0.0596

The underperformance mainly came from missed upside because missed upside was larger than avoided downside, while turnover cost was small but still negative.

## Big-Up Probability Diagnostic
- Lowest big-up-probability bin mean prediction: 0.0847
- Lowest bin realized big-up rate: 0.6364
- Lowest bin mean target return: 0.0380
- Highest big-up-probability bin mean prediction: 0.7021
- Highest bin realized big-up rate: 0.3667
- Highest bin mean target return: 0.0220
- Low-probability but positive-return cases exported: 20
- High-probability but weak-return cases exported: 20

When `pred_big_up_prob` was high, the market did not rise enough to compensate for the upside missed during low-probability/reduced-exposure periods. Low `pred_big_up_prob` did not reliably identify weak future returns, because there were still many positive-return cases in the low-probability group.

## Rule Sensitivity
The current rule appears too sensitive to weak/risk regimes because it spent many days below full exposure while the market often continued rising.

## Top Missed-Upside Days
| date | trade_date | fwd_ret_1d | final_position | position_gap_return | pred_big_up_prob |
| --- | --- | --- | --- | --- | --- |
| 2026-03-23 | 2026-03-24 | 0.056140 | 0.880000 | -0.006737 | 0.561511 |
| 2025-01-13 | 2025-01-14 | 0.054711 | 0.880000 | -0.006565 | 0.468121 |
| 2025-01-06 | 2025-01-07 | 0.036661 | 0.880000 | -0.004399 | 0.511810 |
| 2026-04-07 | 2026-04-08 | 0.036417 | 0.880000 | -0.004370 | 0.521079 |
| 2025-04-09 | 2025-04-10 | 0.035942 | 0.880000 | -0.004313 | 0.551670 |
| 2026-03-05 | 2026-03-06 | 0.034677 | 0.880000 | -0.004161 | 0.309234 |
| 2025-04-08 | 2025-04-09 | 0.034474 | 0.880000 | -0.004137 | 0.544900 |
| 2026-04-03 | 2026-04-07 | 0.033180 | 0.880000 | -0.003982 | 0.465800 |
| 2025-04-30 | 2025-05-06 | 0.031114 | 0.880000 | -0.003734 | 0.523030 |
| 2025-04-11 | 2025-04-14 | 0.026884 | 0.880000 | -0.003226 | 0.546184 |

## Top Successful Defensive Days
| date | trade_date | fwd_ret_1d | final_position | position_gap_return | pred_big_up_prob |
| --- | --- | --- | --- | --- | --- |
| 2026-03-20 | 2026-03-23 | -0.063988 | 0.880000 | 0.007679 | 0.662845 |
| 2025-01-02 | 2025-01-03 | -0.047828 | 0.880000 | 0.005739 | 0.468504 |
| 2026-04-02 | 2026-04-03 | -0.036813 | 0.880000 | 0.004418 | 0.584046 |
| 2026-03-19 | 2026-03-20 | -0.036758 | 0.880000 | 0.004411 | 0.434687 |
| 2025-04-03 | 2025-04-07 | -0.130879 | 0.970000 | 0.003926 | 0.267771 |
| 2025-01-09 | 2025-01-10 | -0.031548 | 0.880000 | 0.003786 | 0.375692 |
| 2026-03-18 | 2026-03-19 | -0.029633 | 0.880000 | 0.003556 | 0.242996 |
| 2025-09-02 | 2025-09-03 | -0.025995 | 0.880000 | 0.003119 | 0.596124 |
| 2025-12-10 | 2025-12-11 | -0.025896 | 0.880000 | 0.003107 | 0.323318 |
| 2026-03-16 | 2026-03-17 | -0.022282 | 0.880000 | 0.002674 | 0.291372 |

## Suggested Next Step
Do not tune blindly. First test whether exposure cuts should be less frequent, whether the severe defensive trigger should require stronger confirmation, and whether low `pred_big_up_prob` should reduce exposure only when downside probability and trend/risk filters agree.
