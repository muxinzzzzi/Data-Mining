# 实验总结

## 配置
- model: LightGBM
- horizon: 20
- feature_top_k: 50
- target_label: avoid_loss_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.3750176225666384
- avg_excess_return_vs_buy_hold: -0.004569465304322266
- positive_fold_ratio: 0.0
- avg_missed_upside: 0.011382251382849555
- avg_avoided_downside: 0.007117996630291234

## 测试期最终表现
- test_total_return: 1.0820838219033684
- test_excess_return_vs_buy_hold: -0.026656861476753058
- test_max_drawdown: -0.17009978105739942
- test_sharpe: 2.326599188210606

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.27464248285785053 | -0.4941571429332826 | 0.5619407062557262 | -0.9448113555690254 | -0.46104760078474205 | -1.0718137174820677 | 0.5128205128205128 | 0.9882222222222222 | 0.947 | 1.0 | 26 | 0.4240000000000004 | 0.0004240000000000004 | 0 | 0 | 0 | nan | 0 | 0 | 0.021604264557703392 | 0.016140204768798183 | -0.0058880597889052096 | -0.002906396930692501 | -0.012681974929949656 | 0.014803619186514908 | -0.8566807055873251 | -0.27173608592715803 | -0.46104760078474205 | False | 0.011777777777777788 | 0.2222222222222222 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.4530693239815402 | 1.0598410893750678 | 0.39356700764820773 | 2.1122426969118737 | -0.16271471376900504 | 6.513492632753863 | 0.576 | 0.989824 | 0.947 | 1.0 | 24 | 0.4240000000000004 | 0.0004240000000000004 | 0 | 0 | 0 | nan | 0 | 0 | 0.012542489590845274 | 0.005213785122075518 | -0.0077527044687697565 | -0.010801998982274297 | -0.015629452209039815 | 0.00624671178832867 | -2.50202870544497 | 0.4638713229638145 | -0.16271471376900504 | False | 0.01017600000000001 | 0.192 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。