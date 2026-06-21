# External Market Overlay Discussion

## Motivation

The external market overlay experiment tests whether broader market information can improve the micro-cap index enhancement strategy. The final main strategy, `ml_risk_budget_enhancement`, does not use the external overlay as its primary signal. The overlay is reported as a supplementary experiment because it helps evaluate a natural extension: whether market-wide risk variables can reduce drawdown without sacrificing too much upside participation.

## External Data Sources

The experiment uses three external data sources:

| File | Role in the overlay |
| --- | --- |
| `index_daily.csv` | Captures broad index trend, drawdown, volatility, liquidity, amount changes, and style-relative strength. |
| `margin_daily.csv` | Captures financing balance and margin-buying behavior, including changes, moving-average gaps, z-scores, and `margin_risk_off`. |
| `shibor_daily.csv` | Captures short-rate and funding-pressure information through rate changes, z-scores, term slopes, and `shibor_tightening`. |

The current feature report records 133 external features in the feature dataframe. `north_money.csv` is present but is excluded under the current project constraint, and `industry_daily.csv` is empty.

## Leakage Control Through `shift(1)`

The external features are computed on their own raw market timeline, shifted by one raw market day, and then backward-aligned to the 880823 trading dates. This `shift(1)` rule is important because it prevents same-day external market information from leaking into a trading decision that would not have been known at decision time.

In report language, the overlay should be described as using lagged external market states. The external risk score does not look ahead to the day being traded.

## External Risk Construction

The composite external risk score combines seven binary conditions:

- Weak broad-index trend.
- Broad-index drawdown.
- Elevated broad-index volatility.
- Weak broad-index liquidity.
- Weak small-cap or growth-style relative strength.
- Margin risk-off condition.
- Shibor tightening condition.

During the test period, the mean external market risk score is 1.6355, with a maximum of 6. The most frequent individual condition is weak index trend, occurring on 43.30% of test observations. Index liquidity risk appears on 29.28%, index volatility risk on 23.99%, style weakness on 21.81%, margin risk on 18.38%, index drawdown on 16.51%, and Shibor tightening on 10.28%.

## Hard Overlay

The hard overlay directly cuts exposure when the external risk score rises. It is useful for stress testing whether external market information contains defensive value. The answer is yes, but the cost is high.

For example, the hard external risk-budget variants improve maximum drawdown relative to buy-and-hold. The `index_margin` hard overlay improves maximum drawdown by 1.77 percentage points, and the `all` hard overlay improves maximum drawdown by 1.77 percentage points. However, these variants also miss substantial upside. The hard `index_margin` variant has missed upside of 21.39% versus avoided downside of 15.61%, and the hard `all` variant has missed upside of 21.83% versus avoided downside of 15.84%. Both produce strongly negative excess returns versus buy-and-hold.

The hard overlay therefore confirms that external features can identify risk-off regimes, but it also shows that aggressive de-risking is costly in a rising market with sharp rebounds.

## Soft Confirmation Overlay

The soft-confirm design makes the external layer less dominant. A watch state does not cut exposure by itself; higher external risk states receive mild weights and require internal stress confirmation before larger exposure reductions. The purpose is to reduce false defensive actions and avoid treating external risk as a standalone sell signal.

The soft-confirm results improve the tradeoff relative to hard overlays. For example, `ml_external_soft_confirm_risk_budget_index_only` has cumulative return 109.19%, excess return -1.69 percentage points, missed upside 8.72%, and avoided downside 8.07%. This is much less costly than the hard overlay, but it still does not beat the original risk-budget strategy. The soft-confirm early-stress index-only variant also reduces maximum drawdown slightly, but its return remains below buy-and-hold.

## Soft-Lite Overlay

The soft-lite overlay is the mildest version. It keeps the same external/internal confirmation logic but uses smaller risk weights and a tighter no-trade band so that small external-risk changes do not create excessive de-risking.

This is the best external overlay by return. `ml_external_soft_lite_risk_budget_index_only` reaches cumulative return 110.61%, only 0.26 percentage points below buy-and-hold and 0.74 percentage points below the original risk-budget strategy. Its missed upside is 7.20% and avoided downside is 7.21%, which is the most balanced missed-upside versus avoided-downside profile among the risk-budget external variants.

Even so, the soft-lite overlay does not improve the final main result. It reduces the opportunity cost of external filters, but the original `ml_risk_budget_enhancement` remains the best return-oriented no-leverage strategy in the current output bundle.

## Missed Upside vs Avoided Downside

The main interpretation of the external overlay experiment is the tradeoff between missed upside and avoided downside. Hard overlays avoid more downside, but they also miss more upside. In the current test period, missed upside usually dominates because the market is strongly upward-trending and rebounds quickly after risk-off signals.

The original risk-budget strategy has missed upside 6.59% and avoided downside 6.94%, a near-balanced tradeoff that still slightly outperforms buy-and-hold. The hard external overlay variants raise both quantities, but missed upside becomes much larger than avoided downside. Soft-confirm and soft-lite designs reduce this imbalance, with soft-lite index-only almost balancing missed upside and avoided downside. However, the remaining transaction cost and small timing drag prevent it from exceeding the original strategy.

The final report should present external market features as a valuable diagnostic and supplementary risk-control experiment, not as an upgrade to the main strategy. They demonstrate that broader market data can reduce certain drawdown measures, but the final design must be very careful not to cut benchmark participation during rebound regimes.

