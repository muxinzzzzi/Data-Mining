# 实验总结

## 配置
- model: RandomForest
- horizon: 20
- feature_top_k: 80
- target_label: big_down_label
- strategy_family: ml_bull_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.5024977900173142
- avg_excess_return_vs_buy_hold: -0.00031915093394230265
- positive_fold_ratio: 0.0
- avg_missed_upside: 0.00023718102743033748
- avg_avoided_downside: 5.431380237893769e-05

## 测试期最终表现
- test_total_return: 1.1122960465560245
- test_excess_return_vs_buy_hold: 0.0035553631759031035
- test_max_drawdown: -0.17260282010543826
- test_sharpe: 2.332324970464172

## 是否跑赢买入持有
- 是

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_bull_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_bull_full_participation_enhancement | -0.27177162710857805 | -0.48997640056552483 | 0.5691300076564825 | -0.9107802949062275 | -0.46106676317329376 | -1.0627016295715188 | 0.5128205128205128 | 0.9994017094017095 | 0.965 | 1.0 | 2 | 0.07000000000000006 | 7.000000000000006e-05 | 0 | 0 | 0 | nan | 0 | 0 | 0.00014234638261570245 | 0.00016294140713681308 | -4.940497547888943e-05 | -3.554118142001794e-05 | -0.0001064107164160722 | 0.00028669610021272643 | -0.3711620644198376 | -0.27173608592715803 | -0.46104760078474205 | True | 0.0005982905982905988 | 0.017094017094017096 | 2024H1 |
| ml_bull_full_participation_enhancement | 0.4629494113434076 | 1.0868866013451228 | 0.39521559617962765 | 2.1397178590303616 | -0.16271471376900504 | 6.679706931040677 | 0.576 | 0.99972 | 0.965 | 1.0 | 1 | 0.07000000000000006 | 7.000000000000006e-05 | 0 | 0 | 0 | nan | 0 | 0 | 0.00056919669967531 | 0.0 | -0.0006391966996753101 | -0.00092191162040689 | -0.0012886205465454297 | 0.0008589121884120588 | -1.5002937016504654 | 0.4638713229638145 | -0.16271471376900504 | True | 0.00028000000000000025 | 0.008 | 2024H2 |
