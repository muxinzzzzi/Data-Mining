# 实验总结

## 配置
- model: LightGBM
- horizon: 20
- feature_top_k: 50
- target_label: big_down_label
- strategy_family: ml_bull_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.5062981322727944
- avg_excess_return_vs_buy_hold: -0.00017147735361366934
- positive_fold_ratio: 0.0
- avg_missed_upside: 0.00018431131227581465
- avg_avoided_downside: 8.654983806264845e-05

## 测试期最终表现
- test_total_return: 1.1078730483783539
- test_excess_return_vs_buy_hold: -0.0008676350017675638
- test_max_drawdown: -0.17260282010543793
- test_sharpe: 2.330169429128056

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_bull_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_bull_full_participation_enhancement | -0.27173608592715803 | -0.4897815582681906 | 0.5691327216655067 | -0.9105889817411069 | -0.46104760078474205 | -1.0623231905654447 | 0.5128205128205128 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | -0.27173608592715803 | -0.46104760078474205 | True | 0.0 | 0.0 | 2024H1 |
| ml_bull_full_participation_enhancement | 0.4633568909029735 | 1.0879870547401902 | 0.3952076357934635 | 2.141178620611512 | -0.16271471376900482 | 6.686470015764724 | 0.576 | 0.999456 | 0.966 | 1.0 | 2 | 0.06800000000000006 | 6.800000000000007e-05 | 0 | 0 | 0 | nan | 0 | 0 | 0.000552933936827444 | 0.00025964951418794535 | -0.0003612844226394987 | -0.000514432060841008 | -0.0007283493960412261 | 0.000915059131027204 | -0.7959588307955727 | 0.4638713229638145 | -0.16271471376900504 | True | 0.0005440000000000004 | 0.016 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。