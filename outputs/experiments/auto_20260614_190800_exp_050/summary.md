# 实验总结

## 配置
- model: RandomForest
- horizon: 20
- feature_top_k: 80
- target_label: avoid_loss_label
- strategy_family: ml_ultra_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.5076364867148453
- avg_excess_return_vs_buy_hold: -0.0001159844431535273
- positive_fold_ratio: 0.0
- avg_missed_upside: 0.00012468118183363934
- avg_avoided_downside: 5.854841986590925e-05

## 测试期最终表现
- test_total_return: 1.1139212702163381
- test_excess_return_vs_buy_hold: 0.005180586836216694
- test_max_drawdown: -0.17260282010543826
- test_sharpe: 2.335398066890825

## 是否跑赢买入持有
- 是

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_ultra_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_ultra_conservative_full_participation_enhancement | -0.27173608592715803 | -0.4897815582681906 | 0.5691327216655067 | -0.9105889817411069 | -0.46104760078474205 | -1.0623231905654447 | 0.5128205128205128 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | -0.27173608592715803 | -0.46104760078474205 | True | 0.0 | 0.0 | 2024H1 |
| ml_ultra_conservative_full_participation_enhancement | 0.4635233696343539 | 1.0876790193870365 | 0.3952220055706167 | 2.1416969981035274 | -0.16271471376900493 | 6.684576914974886 | 0.576 | 0.9996320000000001 | 0.977 | 1.0 | 2 | 0.04600000000000004 | 4.600000000000004e-05 | 0 | 0 | 0 | nan | 0 | 0 | 0.000374043545500918 | 0.00017564525959772774 | -0.0002443982859031903 | -0.0003479533294605819 | -0.0004927069443808377 | 0.0006190105886360503 | -0.7959588307955855 | 0.4638713229638145 | -0.16271471376900504 | True | 0.0003680000000000003 | 0.016 | 2024H2 |
