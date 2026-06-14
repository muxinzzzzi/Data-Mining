# 实验总结

## 配置
- model: RandomForest
- horizon: 20
- feature_top_k: 80
- target_label: avoid_loss_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.097131981061085
- avg_excess_return_vs_buy_hold: -0.006890441374568142
- positive_fold_ratio: 0.0
- avg_missed_upside: 0.019817213307324996
- avg_avoided_downside: 0.013811972242027829

## 测试期最终表现
- test_total_return: 1.0822645681972518
- test_excess_return_vs_buy_hold: -0.02647611518286963
- test_max_drawdown: -0.17009978105739954
- test_sharpe: 2.3303629541853956

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.27380486182913244 | -0.49289817400964353 | 0.5630569297820548 | -0.9374160587838183 | -0.46104760078474205 | -1.06908304732676 | 0.5128205128205128 | 0.9796581196581201 | 0.92 | 1.0 | 77 | 0.49999999999999956 | 0.0004999999999999996 | 0 | 0 | 0 | nan | 0 | 0 | 0.02479007470800628 | 0.020845642837522774 | -0.004444431870483506 | -0.0020687759019744068 | -0.00957262249027221 | 0.011330532759307245 | -0.8448519318219142 | -0.27173608592715803 | -0.46104760078474205 | False | 0.020341880341880343 | 0.6581196581196581 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.4452687747420845 | 1.0394961583079976 | 0.38650262366217736 | 2.115596832952162 | -0.15928675727614816 | 6.525942119004097 | 0.576 | 0.9752000000000004 | 0.92 | 0.98 | 125 | 0.4399999999999996 | 0.00043999999999999964 | 0 | 0 | 0 | nan | 0 | 0 | 0.03466156521396871 | 0.02059027388856071 | -0.014511291325407998 | -0.01860254822173002 | -0.029254763312022532 | 0.009970510910309474 | -2.9341288099663188 | 0.4638713229638145 | -0.16271471376900504 | False | 0.024800000000000013 | 1.0 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。