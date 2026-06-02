# 实验总结

## 配置
- model: ExtraTrees
- horizon: 10
- feature_top_k: 50
- target_label: big_down_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.3715801022976697
- avg_excess_return_vs_buy_hold: -0.005855334351543247
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.016577869833153708
- avg_avoided_downside: 0.01239631454655482

## 测试期最终表现
- test_total_return: 1.0889211379852108
- test_excess_return_vs_buy_hold: -0.019819545394910598
- test_max_drawdown: -0.17060060232575014
- test_sharpe: 2.334709766235647

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.27039255458396394 | -0.48775204509852776 | 0.5636625128709055 | -0.9179511933841275 | -0.46104760078474205 | -1.0579212304072996 | 0.5128205128205128 | 0.9864273504273501 | 0.92 | 1.0 | 70 | 0.5559999999999996 | 0.0005559999999999995 | 0 | 0 | 0 | nan | 0 | 0 | 0.01841994684234526 | 0.019361911800757105 | 0.0003859649584118469 | 0.0013435313431940932 | 0.0008313091411947936 | 0.01283424959493502 | 0.06477271109974876 | -0.27173608592715803 | -0.46104760078474205 | False | 0.013572649572649573 | 0.5982905982905983 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.44496178856599067 | 1.0386229120071921 | 0.3881608038100901 | 2.1071028143533974 | -0.16166192151907055 | 6.424660193616901 | 0.576 | 0.9774079999999997 | 0.92 | 1.0 | 122 | 0.9199999999999994 | 0.0009199999999999994 | 0 | 0 | 0 | nan | 0 | 0 | 0.03131366265711586 | 0.017827031838907357 | -0.014406630818208503 | -0.018909534397823835 | -0.029043767729508344 | 0.010795967951368234 | -2.690242121905101 | 0.4638713229638145 | -0.16271471376900504 | False | 0.022592 | 0.976 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。