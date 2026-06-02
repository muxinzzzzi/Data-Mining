# 实验总结

## 配置
- model: HistGradientBoosting
- horizon: 10
- feature_top_k: 50
- target_label: big_down_label
- strategy_family: ml_ultra_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.400865944015436
- avg_excess_return_vs_buy_hold: -0.003987134934936594
- positive_fold_ratio: 0.0
- avg_missed_upside: 0.008092156617534773
- avg_avoided_downside: 0.004667218390121959

## 测试期最终表现
- test_total_return: 1.0872542955072886
- test_excess_return_vs_buy_hold: -0.021486387872832857
- test_max_drawdown: -0.17260282010543815
- test_sharpe: 2.3086385009852606

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_ultra_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_ultra_conservative_full_participation_enhancement | -0.2735384315311309 | -0.4924973700127996 | 0.5622602076322204 | -0.9381554254302916 | -0.46104760078474205 | -1.0682137141035488 | 0.5128205128205128 | 0.9928888888888888 | 0.94 | 1.0 | 18 | 0.6720000000000006 | 0.0006720000000000006 | 0 | 0 | 0 | nan | 0 | 0 | 0.016791594935198443 | 0.013172908350974035 | -0.004290686584224409 | -0.0018023456039728547 | -0.009241478796791037 | 0.015066298406799001 | -0.6133874789457651 | -0.27173608592715803 | -0.46104760078474205 | False | 0.0071111111111111175 | 0.15384615384615385 | 2024H1 |
| ml_ultra_conservative_full_participation_enhancement | 0.4537122637629776 | 1.0579387366762796 | 0.3943390127505694 | 2.1111369778176425 | -0.1633928439155925 | 6.474816836059251 | 0.576 | 0.9963679999999999 | 0.94 | 1.0 | 10 | 0.5040000000000004 | 0.0005040000000000004 | 0 | 0 | 0 | nan | 0 | 0 | 0.007484874917405877 | 0.0008287468193918429 | -0.007160128098014034 | -0.010159059200836928 | -0.014434818245596273 | 0.004575835063385881 | -3.154575732219521 | 0.4638713229638145 | -0.16271471376900504 | False | 0.0036320000000000033 | 0.08 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。