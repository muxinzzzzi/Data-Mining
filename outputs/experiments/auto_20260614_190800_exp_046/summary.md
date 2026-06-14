# 实验总结

## 配置
- model: RandomForest
- horizon: 10
- feature_top_k: 50
- target_label: big_up_label
- strategy_family: ml_bull_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.5104338880190096
- avg_excess_return_vs_buy_hold: 0.0
- positive_fold_ratio: 0.0
- avg_missed_upside: 0.0
- avg_avoided_downside: 0.0

## 测试期最终表现
- test_total_return: 1.1131672500650112
- test_excess_return_vs_buy_hold: 0.004426566684889721
- test_max_drawdown: -0.17260282010543837
- test_sharpe: 2.3337838878560233

## 是否跑赢买入持有
- 是

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_bull_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_bull_full_participation_enhancement | -0.27173608592715803 | -0.4897815582681906 | 0.5691327216655067 | -0.9105889817411069 | -0.46104760078474205 | -1.0623231905654447 | 0.5128205128205128 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | -0.27173608592715803 | -0.46104760078474205 | True | 0.0 | 0.0 | 2024H1 |
| ml_bull_full_participation_enhancement | 0.4638713229638145 | 1.0870349931267804 | 0.3952527663238839 | 2.1427768810741616 | -0.16271471376900504 | 6.68061890622854 | 0.576 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.4638713229638145 | -0.16271471376900504 | True | 0.0 | 0.0 | 2024H2 |
