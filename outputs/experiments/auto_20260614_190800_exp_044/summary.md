# 实验总结

## 配置
- model: RandomForest
- horizon: 20
- feature_top_k: 80
- target_label: big_down_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.5803332959664982
- avg_excess_return_vs_buy_hold: 0.0003064571798376085
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.017457542006088483
- avg_avoided_downside: 0.01762825322426656

## 测试期最终表现
- test_total_return: 1.0839260589619313
- test_excess_return_vs_buy_hold: -0.024814624418190157
- test_max_drawdown: -0.17009978105739976
- test_sharpe: 2.328156179792638

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23279376974397725 | 0.5025320061868521 | 0.17896597964714242 | 2.466649969270394 | -0.07039174188857789 | 7.139076157289915 | 0.5403225806451613 | 0.9960241935483871 | 0.971 | 1.0 | 17 | 0.29000000000000026 | 0.00029000000000000027 | 0 | 0 | 0 | nan | 0 | 0 | 0.004390644323608726 | 0.002787121909377346 | -0.00189352241423138 | -0.002104557248065575 | -0.0038481261966637724 | 0.0033985087499612157 | -1.1322984519924182 | 0.23489832699204283 | -0.07129994372538018 | False | 0.003975806451612907 | 0.13709677419354838 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.2673019331975628 | -0.48306702189766293 | 0.5593906131290073 | -0.9131574454130589 | -0.45938144092460464 | -1.0515597254547027 | 0.5128205128205128 | 0.9818803418803421 | 0.945 | 1.0 | 48 | 0.37100000000000033 | 0.00037100000000000034 | 0 | 0 | 0 | nan | 0 | 0 | 0.02420040097104788 | 0.02802303200362374 | 0.0034516310325758584 | 0.004434152729595242 | 0.007434282224009538 | 0.016763019185091682 | 0.44349303320139805 | -0.27173608592715803 | -0.46104760078474205 | False | 0.01811965811965813 | 0.41025641025641024 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.46246109902179766 | 1.0869124683080784 | 0.38948246347381144 | 2.1636582782604483 | -0.15792616788199132 | 6.882408931243506 | 0.576 | 0.976512 | 0.945 | 1.0 | 72 | 0.39200000000000035 | 0.0003920000000000004 | 0 | 0 | 0 | nan | 0 | 0 | 0.023781580723608847 | 0.022074605759798596 | -0.0020989749638102508 | -0.0014102239420168416 | -0.0042315335270414626 | 0.009892954801426499 | -0.42773201859077564 | 0.4638713229638145 | -0.16271471376900504 | False | 0.02348800000000002 | 0.576 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。