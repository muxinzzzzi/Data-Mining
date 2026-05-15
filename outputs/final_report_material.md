# Final Report Material

## Research Question
This project uses daily OHLCV data and auxiliary 5-minute intraday features for a micro-cap index to build a machine-learning enhanced index strategy. The investment test period is 2025-01-01 to 2026-05-06, with initial capital of RMB 100,000.

## Leakage Control
The prediction layer uses fixed validation and test periods, no random train/test split, and strict walk-forward training. Every training sample must satisfy `target_end_date < chunk_start`. Feature selection, preprocessing, and model selection are performed using training or validation data only. Trading decisions use signals available after close on date t and apply them to the next tradable return.

## Prediction Layer
The out-of-sample prediction signal is moderate. Ordinary direction AUC and clean-direction AUC remain below 0.60, while big-up AUC is above 0.60. This means the model is better at identifying strong upside opportunities than predicting every normal up/down day.

## Decision Layer
The final no-leverage strategy is benchmark-aware index enhancement. It keeps exposure close to 1.00 in normal and favorable regimes, uses the big-up probability and tail score to maintain full participation in strong upside regimes, and only mildly reduces exposure when downside risk, weak trend, high volatility, or drawdown risk is present. Direct ML timing is included as an interpretable BUY/HOLD/SELL baseline but is not treated as the main result unless it truly outperforms.

## Validation-Selected Parameters
- ml_big_up_index_enhancement: {'model_name': 'ExtraTreesDeep', 'horizon': 10, 'big_up_prob_threshold': 0.56, 'big_down_prob_threshold': 0.58, 'tail_score_threshold': 0.03, 'mild_cut_exposure': 0.97, 'defensive_cut_exposure': 0.92, 'severe_defensive_exposure': 0.88, 'no_trade_band': 0.01, 'high_volatility_multiplier': 1.1, 'drawdown_threshold': 0.06}
- ml_conservative_full_participation_enhancement: {'model_name': 'RandomForestDeep', 'horizon': 20, 'big_up_prob_threshold': 0.56, 'big_down_prob_threshold': 0.58, 'tail_score_threshold': 0.02, 'mild_cut_exposure': 0.97, 'defensive_cut_exposure': 0.96, 'severe_defensive_exposure': 0.94, 'no_trade_band': 0.03, 'high_volatility_multiplier': 1.2, 'drawdown_threshold': 0.1}
- ml_ultra_conservative_full_participation_enhancement: {'model_name': 'RandomForestDeep', 'horizon': 20, 'big_up_prob_threshold': 0.56, 'big_down_prob_threshold': 0.62, 'tail_score_threshold': 0.04, 'mild_cut_exposure': 0.985, 'defensive_cut_exposure': 0.97, 'severe_defensive_exposure': 0.94, 'no_trade_band': 0.05, 'high_volatility_multiplier': 1.2, 'drawdown_threshold': 0.1}
- ml_direct_signal_timing: {'model_name': 'RandomForestDeep', 'horizon': 20, 'big_up_prob_threshold': 0.56, 'big_down_prob_threshold': 0.58, 'tail_score_threshold': 0.0, 'mild_cut_exposure': 0.95, 'defensive_cut_exposure': 0.92, 'severe_defensive_exposure': 0.88, 'no_trade_band': 0.01, 'high_volatility_multiplier': 1.1, 'drawdown_threshold': 0.06}
- ml_big_up_plus_115: {'model_name': 'ExtraTreesDeep', 'horizon': 10, 'big_up_prob_threshold': 0.56, 'big_down_prob_threshold': 0.58, 'tail_score_threshold': 0.03, 'mild_cut_exposure': 0.97, 'defensive_cut_exposure': 0.92, 'severe_defensive_exposure': 0.88, 'no_trade_band': 0.01, 'high_volatility_multiplier': 1.1, 'drawdown_threshold': 0.06}
- ml_big_up_plus_120: {'model_name': 'ExtraTreesDeep', 'horizon': 10, 'big_up_prob_threshold': 0.56, 'big_down_prob_threshold': 0.58, 'tail_score_threshold': 0.03, 'mild_cut_exposure': 0.97, 'defensive_cut_exposure': 0.92, 'severe_defensive_exposure': 0.88, 'no_trade_band': 0.01, 'high_volatility_multiplier': 1.1, 'drawdown_threshold': 0.06}

## Result Table
| strategy | total_return | excess_return_vs_buy_hold | max_drawdown | sharpe | average_position | maximum_position | total_turnover | benchmark_clone |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| buy_hold | 1.108741 | 0.000000 | -0.172603 | 2.325563 | 1.000000 | 1.000000 | 0.000000 | True |
| ml_big_up_index_enhancement | 0.999306 | -0.109435 | -0.168005 | 2.310947 | 0.939875 | 1.000000 | 1.700000 | False |
| ml_big_up_plus_115 | 1.012838 | -0.095902 | -0.168005 | 2.291135 | 0.967531 | 1.150000 | 5.300000 | False |
| ml_big_up_plus_120 | 1.017263 | -0.091478 | -0.168005 | 2.283507 | 0.976750 | 1.200000 | 6.500000 | False |
| ml_conservative_full_participation_enhancement | 1.074227 | -0.034514 | -0.167594 | 2.324654 | 0.983312 | 1.000000 | 0.840000 | False |
| ml_direct_signal_timing | 0.935698 | -0.173043 | -0.138399 | 2.355290 | 0.872000 | 1.000000 | 2.520000 | False |
| ml_ultra_conservative_full_participation_enhancement | 1.086120 | -0.022621 | -0.172603 | 2.312170 | 0.991562 | 1.000000 | 0.120000 | False |

## Reporting Note
No-leverage strategies and enhanced-exposure strategies are separated. Enhanced-exposure strategies allow exposure above 1.00 and are therefore leverage-assisted experiments rather than the primary no-leverage enhanced-index result.
