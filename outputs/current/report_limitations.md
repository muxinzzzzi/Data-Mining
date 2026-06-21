# Report Limitations and Future Work

## Limitations

1. The test period is strongly upward-trending. This makes buy-and-hold a difficult benchmark to beat and makes even mild defensive exposure cuts costly. The main positive excess return is small, so conclusions should be interpreted as evidence of incremental enhancement rather than a large economic edge.

2. The `total_return` field in the result tables is cumulative return in decimal form, not final wealth multiple. This naming can be confusing in report writing and should be clearly defined wherever the tables are used.

3. The validation-best MLP does not generalize to the test period. Its large validation-test gap indicates overfitting risk and limits the claim that neural models provide robust predictive power in this setting.

4. Feature importance for the final `LightGBM / 5D` model cannot be recovered from the current frozen output bundle because no saved model artifact or feature-importance cache is available. The report should avoid fabricating feature importance and should rely on model stability, signal diagnostics, and strategy attribution for interpretation.

5. The strategy uses a single index universe and one final test window. The results may not generalize across market regimes, index definitions, liquidity environments, or longer samples.

6. External market overlays reduce some risk measures but often increase missed upside. This suggests that the external risk score is directionally informative but not yet precise enough to support aggressive exposure cuts.

7. Transaction cost is modeled with a fixed rate. Real implementation may face liquidity constraints, slippage, market impact, limit-up/limit-down effects, and execution timing frictions that are not fully captured.

8. The decision layer uses hand-designed risk-budget rules. Although the rules are validation-selected and interpretable, they may still embed design choices that are sensitive to the sample period.

## Future Work

1. Persist model artifacts and feature-importance tables during the walk-forward workflow so that final report interpretation can include reproducible LightGBM feature importance without retraining.

2. Evaluate the strategy on additional out-of-sample windows, especially sideways and bear-market regimes, to test whether the risk-budget and early-stress components add more value when downside protection matters more.

3. Add multi-fold or rolling-window robustness gates as a formal selection criterion, rather than relying on one validation window for final model and parameter selection.

4. Improve external overlay calibration by learning softer state-dependent weights or by conditioning overlay strength on rebound probability, so that external risk filters avoid excessive missed upside.

5. Compare the current decision-rule approach with constrained portfolio optimization or reinforcement-style allocation methods, while preserving leakage control and no-leverage constraints.

6. Incorporate more realistic execution assumptions, including slippage, liquidity caps, turnover limits, and tradability filters for micro-cap exposure.

7. Expand interpretability analysis beyond feature importance by using signal-bucket diagnostics, event studies, drawdown attribution, and regime-specific performance decomposition.

8. Reframe prediction model selection around downstream utility. Future model selection should jointly evaluate prediction quality, stability, turnover impact, missed upside, avoided downside, and realized excess return.

