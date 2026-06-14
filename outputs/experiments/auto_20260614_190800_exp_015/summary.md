# 实验总结

## 配置
- model: ExtraTrees
- horizon: 10
- feature_top_k: 50
- target_label: big_up_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.379795954911292
- avg_excess_return_vs_buy_hold: -0.0034157643039000782
- positive_fold_ratio: 0.0
- avg_missed_upside: 0.014012059160491938
- avg_avoided_downside: 0.010862460122499884

## 测试期最终表现
- test_total_return: 1.0906761460062162
- test_excess_return_vs_buy_hold: -0.01806453737390523
- test_max_drawdown: -0.17260282010543804
- test_sharpe: 2.316538102992007

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.27345949199838626 | -0.49237858487415576 | 0.563763408836412 | -0.9336724682874226 | -0.46104760078474205 | -1.0679560722929384 | 0.5128205128205128 | 0.9814017094017095 | 0.932 | 1.0 | 32 | 0.33999999999999975 | 0.0003399999999999997 | 0 | 0 | 0 | nan | 0 | 0 | 0.023556625460758415 | 0.020124588192925896 | -0.0037720372678325185 | -0.0017234060712282284 | -0.008124387961485412 | 0.014602630776718537 | -0.5563646774140448 | -0.27173608592715803 | -0.46104760078474205 | False | 0.018598290598290587 | 0.27350427350427353 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.4553474361233425 | 1.0674195498442258 | 0.39198119949556054 | 2.1272680605158576 | -0.1632809940932739 | 6.537316579751253 | 0.576 | 0.9820480000000001 | 0.932 | 1.0 | 33 | 0.47599999999999965 | 0.00047599999999999954 | 0 | 0 | 0 | nan | 0 | 0 | 0.018479552020717396 | 0.012462792174573755 | -0.006492759846143641 | -0.008523886840472006 | -0.013089403849825569 | 0.009678494794929023 | -1.3524214381645032 | 0.4638713229638145 | -0.16271471376900504 | False | 0.01795199999999999 | 0.264 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。