# Report Contributions

1. This project builds a leakage-controlled machine-learning index-enhancement pipeline for the 880823 micro-cap index, combining daily OHLCV data, intraday 5-minute features, external market variables, fixed validation/test splits, and walk-forward prediction.

2. The study distinguishes prediction-layer performance from strategy-layer utility. It shows that the validation-best MLP model suffers a large validation-test gap, while the final `LightGBM / 5D` strategy is selected because it delivers better realized trading utility inside a risk-budget decision rule.

3. The project proposes a high-participation risk-budget enhancement design that treats ML signals as auxiliary risk and recovery inputs rather than direct buy/sell commands. This design preserves benchmark exposure in normal regimes and avoids aggressive all-in/all-out timing.

4. The analysis introduces an early-stress risk-budget variant that combines short-term downside acceleration, intraday selling pressure, liquidity stress, weak trend, and volatility expansion. This variant slightly improves maximum drawdown, illustrating how intraday and stress-oriented features can support defensive timing.

5. The report evaluates external market overlays using broad index, margin, and Shibor data with explicit `shift(1)` leakage control. The hard, soft-confirm, and soft-lite overlay experiments document the central tradeoff between avoided downside and missed upside in a strongly rising test period.

