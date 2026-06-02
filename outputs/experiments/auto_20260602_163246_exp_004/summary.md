# 实验总结

## 配置
- model: LightGBM
- horizon: 5
- feature_top_k: 50
- target_label: reduce_position_worth_label
- strategy_family: ml_ultra_conservative_full_participation_enhancement

## 验证结果
- robust_score: 1.4562049240013908
- avg_excess_return_vs_buy_hold: 0.0002512734045230225
- positive_fold_ratio: 0.3333333333333333

## 测试结果
- test_total_return: 1.1133922826356648
- test_excess_return_vs_buy_hold: 0.0046515992555433705
- test_max_drawdown: -0.17260282010543793
- test_sharpe: 2.336795478601378

## 是否跑赢买入持有
- 是

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_ultra_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_ultra_conservative_full_participation_enhancement | -0.27173608592715803 | -0.4897815582681906 | 0.5691327216655068 | -0.9105889817411068 | -0.46104760078474205 | -1.0623231905654447 | 0.5128205128205128 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | -0.27173608592715803 | -0.46104760078474205 | True | 0.0 | 0.0 | 2024H1 |
| ml_ultra_conservative_full_participation_enhancement | 0.46462514317738357 | 1.092425402120603 | 0.3947455651990798 | 2.147651967747167 | -0.1627147137690047 | 6.713746881375749 | 0.576 | 0.9985599999999999 | 0.955 | 1.0 | 4 | 0.09000000000000008 | 9.000000000000009e-05 | 0 | 0 | 0 | nan | 0 | 0 | 0.0015996566274583514 | 0.00210513281785767 | 0.00041547619039931856 | 0.0007538202135690675 | 0.0008375999998450189 | 0.0030502394275929118 | 0.2746013943259558 | 0.4638713229638145 | -0.16271471376900504 | False | 0.0014400000000000012 | 0.032 | 2024H2 |
