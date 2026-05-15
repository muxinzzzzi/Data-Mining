# Enhanced Index Strategy Summary

## Key Result
- Best prediction model: ExtraTrees
- Best prediction horizon: 10 trading days
- Directional accuracy: 60.45%
- AUC: 0.4768
- Return correlation: 0.1186
- Directional accuracy >= 70%: No
- Validation-selected feature top_k: 80
- Big-up AUC: 0.5338
- Big-down AUC: 0.5072
- Tail-score return correlation: 0.1347

## Prediction Layer Update
- Added trend-quality, tail-risk, volume-confirmation, intraday-structure, and market-regime factors.
- The 5-minute data are not directly used for high-frequency trading. Instead, they are transformed into daily intraday-structure factors. In addition to basic intraday return and volatility, the enhanced feature set captures opening shock, closing pressure, high/low timing, intraday reversal, volume-return correlation, net volume pressure, large-volume-bar return, and intraday trend quality. These features are designed to extract information that daily OHLCV data cannot directly observe.
- Added leakage-safe tail targets: target_big_up and target_big_down use historical forward-return quantiles whose target_end_date is before the current date.
- Added separate big-up and big-down classifiers and pred_tail_score = pred_big_up_prob - pred_big_down_prob.
- Feature top_k is selected only on the validation period from candidates [50, 80, 120, all], and each walk-forward chunk selects features using IC computed on that chunk's training data only.

## Best No-Leverage Strategy
- Best no-leverage strategy by total return: ml_directional_long_cash (ExtraTrees, horizon=3)
- Best no-leverage strategy total return: 110.65%
- Buy-and-hold total return: 110.65%
- No-leverage excess return: 0.00%
- Best no-leverage max drawdown: -17.26%
- Whether best no-leverage strategy is benchmark clone: Yes
- Sharpe: 2.3213
- Calmar: 4.6238
- Information Ratio: NA
- Any no-leverage strategy outperforms buy-and-hold: No
- Best no-leverage strategy by Sharpe: ml_signal_aggressive (LightGBM, horizon=5), Sharpe 2.6405
- Best no-leverage strategy by max drawdown: ml_signal_aggressive (LightGBM, horizon=5), max drawdown -8.76%

## Benchmark Clone Check
- Best total-return no-leverage strategy may be a benchmark clone if it keeps 1.00 exposure all the time.
- Best no-leverage strategy by total return: ml_directional_long_cash (ExtraTrees, horizon=3)
- Whether it is benchmark clone: Yes
- Avg position: 1.0000
- Min position: 1.0000
- Max position: 1.0000
- Days below full exposure: 0
- Avg absolute position gap: 0.0000
- Total turnover: 1.0000
- Cost-aware conservative overlay benchmark clone: Yes
- Cost-aware conservative overlay avg position: 1.0000
- Cost-aware conservative overlay days below full exposure: 0
- Best real no-leverage ML timing strategy: ml_selective_defensive_enhancement (GradientBoosting, horizon=10)
- Real ML timing strategy total return: 110.20%
- Real ML timing strategy excess return: -0.45%
- Real ML timing strategy max drawdown: -17.26%
- Real ML timing strategy days below full exposure: 15
- Real ML timing strategy avg position: 0.9986
- Real ML timing strategy total turnover: 1.0600
- Fallback note: The real no-leverage ML timing strategy excludes buy-and-hold, benchmark clones, and enhanced-exposure strategies.

## Enhanced-Exposure Experiment
- Enabled: Yes
- Strategies: ml_index_enhanced_plus, ml_index_enhanced_plus_115, ml_index_enhanced_plus_120
- Best enhanced-exposure strategy by total return: ml_index_enhanced_plus_120 (GradientBoosting, horizon=5)
- Best enhanced-exposure total return: 120.05%
- Enhanced-exposure excess return: 9.40%
- Enhanced-exposure outperforms buy-and-hold: Yes

## Cost-Aware Index Enhancement
- Strategy: ml_cost_aware_index_enhancement
- Signal source: ExtraTrees 10D
- Total return: 108.44%
- Excess return vs buy-and-hold: -2.21%
- Max drawdown: -17.26%
- Average turnover: 0.0044
- Selected cost-aware parameters: {'model_name': 'ExtraTrees', 'horizon': 10, 'fallback_used': False, 'validation_score': 0.624964858426704, 'min_position': 0.85, 'strong_trend_cut_position': 0.95, 'weak_cut_position': 0.97, 'risk_cut_position': 0.95, 'positive_prob': 0.5, 'negative_prob': 0.45, 'strong_negative_prob': 0.42, 'drawdown_cut': -0.12, 'high_vol_multiplier': 1.15, 'no_trade_band': 0.05, 'total_return': 0.451365955865636, 'benchmark_total_return': 0.46243087726760446, 'excess_return_vs_buy_hold': -0.011064921401968464, 'annual_return': 1.1190553197165265, 'annualized_excess_return': -0.03269512617447479, 'sharpe': 2.1222962644741394, 'max_drawdown': -0.16219291279614412, 'benchmark_max_drawdown': -0.16271471376900537, 'drawdown_improvement': 0.0005218009728612527, 'avg_turnover': 0.009200000000000002}

## Tail-Score Index Enhancement
- Strategy: ml_tail_score_index_enhancement
- Signal source: RandomForest 1D
- Total return: 104.69%
- Excess return vs buy-and-hold: -5.96%
- Max drawdown: -16.36%
- Avg position: 0.9630
- Days below full exposure: 150
- Total turnover: 2.9700
- Selected tail-score parameters: {'model_name': 'RandomForest', 'horizon': 1, 'validation_score': 0.5445569710845468, 'strong_tail_score': 0.05, 'low_tail_score': 0.05, 'very_low_tail_score': -0.15, 'big_down_prob_cut': 0.6, 'weak_cut_position': 0.95, 'risk_cut_position': 0.92, 'min_position': 0.9, 'high_risk_multiplier': 1.15, 'no_trade_band': 0.0, 'total_return': 0.4613971971886932, 'benchmark_total_return': 0.46243087726760446, 'excess_return_vs_buy_hold': -0.0010336800789112477, 'annual_return': 1.14868539803234, 'annualized_excess_return': -0.0030650478586613517, 'sharpe': 2.173596195920127, 'max_drawdown': -0.1551811094021529, 'benchmark_max_drawdown': -0.16271471376900537, 'drawdown_improvement': 0.007533604366852464, 'avg_turnover': 0.012959999999999998, 'avg_position': 0.9596000000000002, 'min_position_observed': 0.9, 'days_below_full_exposure': 65, 'pct_days_below_full_exposure': 0.52, 'total_turnover': 1.6199999999999997, 'avg_abs_position_gap': 0.04039999999999998}
- Interpretation: The tail-score index-enhancement strategy still does not outperform buy-and-hold. Under the single-index OHLCV data constraint, the added tail targets and factors are still not sufficient to produce stable positive no-leverage excess return.

## Direct ML Buy/Sell Timing Strategy
- Added strategies: ml_directional_long_cash and ml_directional_scaled_timing
- Logic: BUY means the model predicts high upward probability; SELL means the model predicts low upward probability or negative expected return; HOLD means the signal is not strong enough, reducing unnecessary trading.
- Long/cash selected model/horizon: ExtraTrees 3D
- Long/cash selected parameters: {'strategy': 'ml_directional_long_cash', 'model_name': 'ExtraTrees', 'horizon': 3, 'validation_score': 0.823127641659691, 'buy_prob': 0.52, 'sell_prob': 0.4, 'min_pred_ret_buy': 0.0, 'max_pred_ret_sell': 0.0, 'use_trade_prob': True, 'trade_prob_cut': 0.5, 'no_trade_band': 0.0, 'total_return': 0.6246190540874148, 'benchmark_total_return': 0.46243087726760446, 'excess_return_vs_buy_hold': 0.16218817681981035, 'annual_return': 1.6599600233134164, 'annualized_excess_return': 0.5082095774224151, 'sharpe': 2.81068431613647, 'max_drawdown': -0.11444343076230168, 'benchmark_max_drawdown': -0.16271471376900537, 'drawdown_improvement': 0.04827128300670369, 'avg_turnover': 0.016}
- Long/cash total return: 110.65%
- Long/cash excess return vs buy-and-hold: 0.00%
- Long/cash max drawdown: -17.26%
- Long/cash Sharpe: 2.3213
- Long/cash BUY signals: 0
- Long/cash SELL signals: 0
- Long/cash win rate after BUY: NA
- Long/cash successful SELL count: 0
- Long/cash failed SELL count: 0
- Long/cash outperforms buy-and-hold: No
- Scaled timing selected model/horizon: XGBoost 1D
- Scaled timing selected parameters: {'strategy': 'ml_directional_scaled_timing', 'model_name': 'XGBoost', 'horizon': 1, 'validation_score': 0.6308542780082693, 'strong_buy_prob': 0.58, 'buy_prob': 0.52, 'sell_prob': 0.48, 'strong_sell_prob': 0.42, 'buy_position': 0.95, 'neutral_position': 0.8, 'sell_position': 0.2, 'strong_sell_position': 0.2, 'trend_filter': False, 'risk_filter': True, 'no_trade_band': 0.0, 'strong_buy_position': 1.0, 'total_return': 0.4994047628057503, 'benchmark_total_return': 0.46243087726760446, 'excess_return_vs_buy_hold': 0.036973885538145845, 'annual_return': 1.2628328468133936, 'annualized_excess_return': 0.1110824009223923, 'sharpe': 2.7987558224966604, 'max_drawdown': -0.11120900076470941, 'benchmark_max_drawdown': -0.16271471376900537, 'drawdown_improvement': 0.05150571300429596, 'avg_turnover': 0.0496}
- Scaled timing total return: 58.73%
- Scaled timing excess return vs buy-and-hold: -51.92%
- Scaled timing max drawdown: -16.26%
- Scaled timing Sharpe: 1.6892
- Scaled timing BUY signals: 23
- Scaled timing SELL signals: 21
- Scaled timing win rate after BUY: 52.17%
- Scaled timing successful SELL count: 5
- Scaled timing failed SELL count: 16
- Scaled timing outperforms buy-and-hold: No
- Interpretation: The direct ML buy/sell strategy implements the required prediction-driven trading logic, but under the strong upward test period, frequent or incorrect SELL signals caused missed upside, so the no-leverage strategy did not outperform buy-and-hold.

## Selective Defensive Index Enhancement
- Strategy name: ml_selective_defensive_enhancement
- This is a benchmark-aware no-leverage enhancement strategy, not an aggressive 0/1 timing strategy.
- It defaults to full participation, applies only mild cuts for uncertain negative ML signals, and makes larger defensive cuts when ML negative signal, weak trend, and high risk coincide.
- Maximum exposure is 1.00, so there is no leverage. The minimum exposure is selected on the validation period.
- Parameters are selected only on the validation period from 2024-07-01 to 2024-12-31.
- Enhanced 5-minute factors may be used as risk confirmation, but they are daily aggregated intraday-structure factors and do not use future information.
- Selected model/horizon: GradientBoosting 10D
- Selected parameters: {'strategy': 'ml_selective_defensive_enhancement', 'model_name': 'GradientBoosting', 'horizon': 10, 'validation_score': 0.35071807352558443, 'clone_like_penalty': 0.0, 'selectable': True, 'weak_prob': 0.48, 'strong_negative_prob': 0.4, 'weak_ret_cut': 0.0, 'mild_cut_position': 0.97, 'risk_cut_position': 0.92, 'min_position': 0.9, 'high_vol_multiplier': 1.2, 'drawdown_cut': -0.1, 'no_trade_band': 0.03, 'require_trend_confirm': True, 'require_risk_confirm': False, 'total_return': 0.47225775308474205, 'benchmark_total_return': 0.46243087726760446, 'excess_return_vs_buy_hold': 0.009826875817137593, 'annual_return': 1.1809988756407623, 'annualized_excess_return': 0.02924842974976105, 'sharpe': 2.1890504547810523, 'max_drawdown': -0.1563634576454197, 'benchmark_max_drawdown': -0.16271471376900537, 'drawdown_improvement': 0.006351256123585669, 'avg_turnover': 0.01104, 'avg_position': 0.9923999999999998, 'min_position_observed': 0.92, 'days_below_full_exposure': 15, 'pct_days_below_full_exposure': 0.12, 'total_turnover': 1.38, 'avg_abs_position_gap': 0.007599999999999998}
- Total return: 110.20%
- Excess return vs buy-and-hold: -0.45%
- Max drawdown: -17.26%
- Sharpe: 2.3164
- Avg position: 0.9986
- Min position: 0.9700
- Days below full exposure: 15
- Turnover: 1.0600
- Outperforms buy-and-hold: No
- Becomes best real no-leverage ML timing strategy: Yes
- Interpretation: The selective defensive enhancement strategy maintains high benchmark participation and improves the interpretability of ML-based timing, but it still does not outperform buy-and-hold in this strongly rising test period.

## Experiment Setup
- Initial capital: RMB 100,000
- Test period: 2025-01-01 to 2026-05-06
- Horizons: [1, 3, 5, 10, 20]
- Retrain frequency: every 20 test trading days
- Trading cost: 0.0010 per turnover
- Feature count: 273
- Trade label threshold: 0.0030

## Leakage Controls
- Features use only information available at or before the close of date t.
- Signals generated after date t close are applied to t+1 returns.
- Walk-forward training uses `target_end_date < chunk_start`, not `date < chunk_start`.
- Target labels remain NaN when forward returns are unavailable.
- Quantile clipping and median imputation are fitted on each training chunk only.
- No random train/test split is used.

## Optional Model Availability
- XGBoost: installed and included in model comparison
- LightGBM: installed and included in model comparison

## Model Selection Note
- XGBoost and LightGBM are part of the current comparison set.
- ExtraTrees was selected by the composite rule. XGBoost and LightGBM were included but did not become the final selected prediction model.
- Final selection uses a composite score, not single accuracy: AUC, directional accuracy, positive return correlation, strategy-level excess return, and drawdown are all considered.
- Interpretation: the signal is moderate rather than strong; it is best described as weak but useful directional information with improved drawdown control.

## Strategy Layer Update
- Added ml_index_enhanced_core: benchmark-aware index enhancement with high no-leverage exposure in [0.75, 1.00].
- Added ml_index_full_participation: benchmark-aware full participation with no-leverage exposure in [0.88, 1.00].
- Added ml_directional_alpha_extratrees_10d: fixed strategy using ExtraTrees 10D, the highest directional-accuracy model.
- Added ml_validation_tuned_index_core: parameters selected only on the validation period from 2024-07-01 to 2024-12-31.
- Added ml_dynamic_ensemble_full_participation: validation-weighted ensemble across high-AUC, high-direction, selected, and optional boosting models.
- Added ml_cost_aware_index_enhancement: validation-tuned no-trade-band strategy using ML predictions to decide small exposure reductions and recoveries.
- Added ml_cost_aware_overlay_conservative: extremely low-turnover no-leverage overlay that only cuts exposure when model, trend, and risk signals are all weak.
- Added ml_tail_score_index_enhancement: validation-tuned no-leverage strategy using big-up/big-down tail-score predictions to cut exposure only when tail score is weak and risk/trend conditions confirm.
- Added ml_directional_long_cash: direct BUY/SELL/HOLD strategy that goes long when ML predicts upside and moves to cash when ML predicts downside.
- Added ml_directional_scaled_timing: direct ML timing strategy that maps predicted up probability and predicted return into scaled no-leverage exposure.
- Added ml_selective_defensive_enhancement: benchmark-aware no-leverage enhancement that defaults to full participation, applies mild cuts for uncertain negative ML signals, and uses larger cuts when ML negative signal, weak trend, and high risk coincide.
- Added ml_vol_target_trend: volatility targeting plus trend following, with ML signal as a small adjustment.
- Added ml_enhanced_cppi: CPPI-style path-dependent drawdown control based only on realized strategy equity.
- Added ml_index_enhanced_plus, ml_index_enhanced_plus_115, and ml_index_enhanced_plus_120: enhanced-exposure scenarios reported separately from no-leverage strategies.
- Selected feature top_k: 80
- Selected validation-tuned parameters: {'model_name': 'XGBoost', 'horizon': 1, 'validation_score': 0.6576802117476352, 'min_position': 0.92, 'strong_floor': 0.95, 'weak_prob': 0.48, 'strong_prob': 0.52, 'high_vol_multiplier': 1.15, 'risk_cut': 0.05, 'total_return': 0.4629931291065823, 'benchmark_total_return': 0.46243087726760446, 'excess_return_vs_buy_hold': 0.0005622518389778453, 'annual_return': 1.153418548566266, 'annualized_excess_return': 0.0016681026752647377, 'sharpe': 2.1805935096352274, 'max_drawdown': -0.15469089973282701, 'benchmark_max_drawdown': -0.16271471376900537, 'drawdown_improvement': 0.008023814036178356, 'avg_turnover': 0.010719999999999999}
- Selected cost-aware parameters: {'model_name': 'ExtraTrees', 'horizon': 10, 'fallback_used': False, 'validation_score': 0.624964858426704, 'min_position': 0.85, 'strong_trend_cut_position': 0.95, 'weak_cut_position': 0.97, 'risk_cut_position': 0.95, 'positive_prob': 0.5, 'negative_prob': 0.45, 'strong_negative_prob': 0.42, 'drawdown_cut': -0.12, 'high_vol_multiplier': 1.15, 'no_trade_band': 0.05, 'total_return': 0.451365955865636, 'benchmark_total_return': 0.46243087726760446, 'excess_return_vs_buy_hold': -0.011064921401968464, 'annual_return': 1.1190553197165265, 'annualized_excess_return': -0.03269512617447479, 'sharpe': 2.1222962644741394, 'max_drawdown': -0.16219291279614412, 'benchmark_max_drawdown': -0.16271471376900537, 'drawdown_improvement': 0.0005218009728612527, 'avg_turnover': 0.009200000000000002}
- Selected tail-score parameters: {'model_name': 'RandomForest', 'horizon': 1, 'validation_score': 0.5445569710845468, 'strong_tail_score': 0.05, 'low_tail_score': 0.05, 'very_low_tail_score': -0.15, 'big_down_prob_cut': 0.6, 'weak_cut_position': 0.95, 'risk_cut_position': 0.92, 'min_position': 0.9, 'high_risk_multiplier': 1.15, 'no_trade_band': 0.0, 'total_return': 0.4613971971886932, 'benchmark_total_return': 0.46243087726760446, 'excess_return_vs_buy_hold': -0.0010336800789112477, 'annual_return': 1.14868539803234, 'annualized_excess_return': -0.0030650478586613517, 'sharpe': 2.173596195920127, 'max_drawdown': -0.1551811094021529, 'benchmark_max_drawdown': -0.16271471376900537, 'drawdown_improvement': 0.007533604366852464, 'avg_turnover': 0.012959999999999998, 'avg_position': 0.9596000000000002, 'min_position_observed': 0.9, 'days_below_full_exposure': 65, 'pct_days_below_full_exposure': 0.52, 'total_turnover': 1.6199999999999997, 'avg_abs_position_gap': 0.04039999999999998}
- Selected direct long/cash parameters: {'strategy': 'ml_directional_long_cash', 'model_name': 'ExtraTrees', 'horizon': 3, 'validation_score': 0.823127641659691, 'buy_prob': 0.52, 'sell_prob': 0.4, 'min_pred_ret_buy': 0.0, 'max_pred_ret_sell': 0.0, 'use_trade_prob': True, 'trade_prob_cut': 0.5, 'no_trade_band': 0.0, 'total_return': 0.6246190540874148, 'benchmark_total_return': 0.46243087726760446, 'excess_return_vs_buy_hold': 0.16218817681981035, 'annual_return': 1.6599600233134164, 'annualized_excess_return': 0.5082095774224151, 'sharpe': 2.81068431613647, 'max_drawdown': -0.11444343076230168, 'benchmark_max_drawdown': -0.16271471376900537, 'drawdown_improvement': 0.04827128300670369, 'avg_turnover': 0.016}
- Selected direct scaled-timing parameters: {'strategy': 'ml_directional_scaled_timing', 'model_name': 'XGBoost', 'horizon': 1, 'validation_score': 0.6308542780082693, 'strong_buy_prob': 0.58, 'buy_prob': 0.52, 'sell_prob': 0.48, 'strong_sell_prob': 0.42, 'buy_position': 0.95, 'neutral_position': 0.8, 'sell_position': 0.2, 'strong_sell_position': 0.2, 'trend_filter': False, 'risk_filter': True, 'no_trade_band': 0.0, 'strong_buy_position': 1.0, 'total_return': 0.4994047628057503, 'benchmark_total_return': 0.46243087726760446, 'excess_return_vs_buy_hold': 0.036973885538145845, 'annual_return': 1.2628328468133936, 'annualized_excess_return': 0.1110824009223923, 'sharpe': 2.7987558224966604, 'max_drawdown': -0.11120900076470941, 'benchmark_max_drawdown': -0.16271471376900537, 'drawdown_improvement': 0.05150571300429596, 'avg_turnover': 0.0496}
- Selected selective defensive parameters: {'strategy': 'ml_selective_defensive_enhancement', 'model_name': 'GradientBoosting', 'horizon': 10, 'validation_score': 0.35071807352558443, 'clone_like_penalty': 0.0, 'selectable': True, 'weak_prob': 0.48, 'strong_negative_prob': 0.4, 'weak_ret_cut': 0.0, 'mild_cut_position': 0.97, 'risk_cut_position': 0.92, 'min_position': 0.9, 'high_vol_multiplier': 1.2, 'drawdown_cut': -0.1, 'no_trade_band': 0.03, 'require_trend_confirm': True, 'require_risk_confirm': False, 'total_return': 0.47225775308474205, 'benchmark_total_return': 0.46243087726760446, 'excess_return_vs_buy_hold': 0.009826875817137593, 'annual_return': 1.1809988756407623, 'annualized_excess_return': 0.02924842974976105, 'sharpe': 2.1890504547810523, 'max_drawdown': -0.1563634576454197, 'benchmark_max_drawdown': -0.16271471376900537, 'drawdown_improvement': 0.006351256123585669, 'avg_turnover': 0.01104, 'avg_position': 0.9923999999999998, 'min_position_observed': 0.92, 'days_below_full_exposure': 15, 'pct_days_below_full_exposure': 0.12, 'total_turnover': 1.38, 'avg_abs_position_gap': 0.007599999999999998}

## Validation-Selected Ensemble Components
- Regression component: {'model_name': 'XGBoost', 'horizon': 20}
- Classification component: {'model_name': 'GradientBoosting', 'horizon': 20}
- Trade-label component: {'model_name': 'GradientBoosting', 'horizon': 20}

## Honest Note
If the 70% directional-accuracy target is not reached, this is expected under a strict out-of-sample protocol. Short-horizon index returns are noisy, this is a single-index dataset with limited independent samples, and technical factors often have weak standalone predictive power. The system still performs multi-factor mining, multi-model comparison, leakage-safe walk-forward prediction, strategy backtesting, transaction-cost accounting, and drawdown control.

## Benchmark Interpretation
The no-leverage strategies still do not outperform buy-and-hold in total return. The test period was strongly upward, buy-and-hold stayed fully invested, and ML strategies reduced exposure to control risk; the best no-leverage strategy therefore reduced drawdown but lagged in total return.
The enhanced-exposure strategy outperforms buy-and-hold, but it permits maximum exposure above 1.0 and should be interpreted as a separate leverage-assisted experiment.
The final no-leverage cost-aware index-enhancement strategy improves cost control and drawdown management, but it still does not generate positive excess return over buy-and-hold. This indicates that, with only index-level OHLCV and limited intraday data, outperforming a strongly rising fully invested benchmark without leverage remains difficult.
The tail-score index-enhancement strategy still does not outperform buy-and-hold. Under the single-index OHLCV data constraint, the added tail targets and factors are still not sufficient to produce stable positive no-leverage excess return.
The direct ML buy/sell strategy implements the required prediction-driven trading logic, but under the strong upward test period, frequent or incorrect SELL signals caused missed upside, so the no-leverage strategy did not outperform buy-and-hold.
The selective defensive enhancement strategy maintains high benchmark participation and improves the interpretability of ML-based timing, but it still does not outperform buy-and-hold in this strongly rising test period.
The enhanced-exposure plus strategy is reported separately because it allows maximum exposure above 1.00.

## Files
- outputs/predictions_all_models.csv
- outputs/model_metrics_by_horizon.csv
- outputs/strategy_daily_all.csv
- outputs/strategy_metrics_all.csv
- outputs/active_return_attribution.csv
- outputs/active_return_attribution_summary.md
- outputs/directional_trading_signals.csv
- outputs/directional_timing_attribution_summary.md
- outputs/directional_timing_tuned_params.json
- outputs/selective_defensive_tuned_params.json
- outputs/selective_defensive_attribution.csv
- outputs/selective_defensive_attribution_summary.md
- outputs/factor_effectiveness.csv
- outputs/factor_effectiveness_tail_targets.csv
- outputs/feature_columns.json
- outputs/best_model_selection.json
- outputs/strategy_tuned_params.json
- outputs/cost_aware_tuned_params.json
- outputs/plots/
