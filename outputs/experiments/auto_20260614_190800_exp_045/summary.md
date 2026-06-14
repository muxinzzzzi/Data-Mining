# 实验总结

## 配置
- model: ExtraTrees
- horizon: 10
- feature_top_k: 50
- target_label: big_down_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.3132922229400363
- avg_excess_return_vs_buy_hold: -0.006238389781989762
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.022365832012706574
- avg_avoided_downside: 0.017576648837673837

## 测试期最终表现
- test_total_return: 1.0848281429777602
- test_excess_return_vs_buy_hold: -0.023912540402361238
- test_max_drawdown: -0.1700997810573993
- test_sharpe: 2.324190192795836

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.2692246242736638 | -0.4859842842494112 | 0.5616365570318738 | -0.9171993590022947 | -0.46104760078474205 | -1.0540870040798929 | 0.5128205128205128 | 0.9809999999999998 | 0.932 | 1.0 | 70 | 0.36899999999999944 | 0.0003689999999999994 | 0 | 0 | 0 | nan | 0 | 0 | 0.024068040680648814 | 0.025882499256921265 | 0.001445458576272452 | 0.0025114616534942513 | 0.0031132953950483155 | 0.0132835531623975 | 0.23437218619046113 | -0.27173608592715803 | -0.46104760078474205 | False | 0.019000000000000003 | 0.5982905982905983 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.44264469196435097 | 1.031201908459534 | 0.3839600266502315 | 2.117505367994205 | -0.15969668518170432 | 6.457253056231088 | 0.576 | 0.9687199999999998 | 0.932 | 1.0 | 122 | 0.6339999999999991 | 0.000633999999999999 | 0 | 0 | 0 | nan | 0 | 0 | 0.04302945535747091 | 0.026847447256100252 | -0.016816008101370655 | -0.021226630999463536 | -0.03390107233236321 | 0.012474232404849019 | -2.7176880494213886 | 0.4638713229638145 | -0.16271471376900504 | False | 0.03128000000000001 | 0.976 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。