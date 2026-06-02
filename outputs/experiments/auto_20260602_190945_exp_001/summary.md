# 实验总结

## 配置
- model: ExtraTrees
- horizon: 10
- feature_top_k: 50
- target_label: avoid_loss_label
- strategy_family: ml_bull_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.7042203901154998
- avg_excess_return_vs_buy_hold: 0.00011047725537360886
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.0006309257131255822
- avg_avoided_downside: 0.0010398141819734027

## 测试期最终表现
- test_total_return: 1.1131672500650112
- test_excess_return_vs_buy_hold: 0.004426566684889721
- test_max_drawdown: -0.17260282010543837
- test_sharpe: 2.3337838878560246

## 是否跑赢买入持有
- 是

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_bull_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_bull_full_participation_enhancement | -0.27040372440114113 | -0.48792287112256627 | 0.5689222492499172 | -0.904230033761078 | -0.46005381800260725 | -1.0605778107460486 | 0.5128205128205128 | 0.9964273504273504 | 0.962 | 1.0 | 11 | 0.07600000000000007 | 7.600000000000007e-05 | 0 | 0 | 0 | nan | 0 | 0 | 0.0012747921511578387 | 0.0031194425459202084 | 0.0017686503947623696 | 0.0013323615260169008 | 0.003809400850257413 | 0.0022223619026996017 | 1.7141226393549875 | -0.27173608592715803 | -0.46104760078474205 | False | 0.003572649572649576 | 0.09401709401709402 | 2024H1 |
| ml_bull_full_participation_enhancement | 0.46287039320391843 | 1.0868738822969752 | 0.39521249686176735 | 2.139455161052054 | -0.16271471376900537 | 6.679628763259441 | 0.576 | 0.999696 | 0.962 | 1.0 | 1 | 0.07600000000000007 | 7.600000000000007e-05 | 0 | 0 | 0 | nan | 0 | 0 | 0.000617984988218908 | 0.0 | -0.0006939849882189081 | -0.0010009297598960742 | -0.0013990737362493178 | 0.0009325332331330875 | -1.500293701650467 | 0.4638713229638145 | -0.16271471376900504 | True | 0.0003040000000000003 | 0.008 | 2024H2 |
