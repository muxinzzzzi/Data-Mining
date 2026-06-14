# 实验总结

## 配置
- model: HistGradientBoosting
- horizon: 20
- feature_top_k: 50
- target_label: big_down_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.423765525084705
- avg_excess_return_vs_buy_hold: -0.002195965246694039
- positive_fold_ratio: 0.0
- avg_missed_upside: 0.0109289841979397
- avg_avoided_downside: 0.008308496481071466

## 测试期最终表现
- test_total_return: 1.066355657732454
- test_excess_return_vs_buy_hold: -0.04238502564766744
- test_max_drawdown: -0.16776120465521194
- test_sharpe: 2.325436875995354

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.27405903738994686 | -0.4932803845001994 | 0.5608339327556044 | -0.9447159083607005 | -0.46104760078474205 | -1.069912051728703 | 0.5128205128205128 | 0.9860000000000001 | 0.937 | 1.0 | 26 | 0.37799999999999967 | 0.00037799999999999965 | 0 | 0 | 0 | nan | 0 | 0 | 0.024185233045970284 | 0.019185526423288374 | -0.005377706622681909 | -0.002322951462788825 | -0.011582752725776439 | 0.017385498607715617 | -0.6662306895607848 | -0.27173608592715803 | -0.46104760078474205 | False | 0.013999999999999988 | 0.2222222222222222 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.4596063786865212 | 1.0747948223684394 | 0.39396135829239304 | 2.1335446510347853 | -0.16205763806051132 | 6.632176275252868 | 0.576 | 0.9914319999999999 | 0.937 | 1.0 | 17 | 0.3149999999999997 | 0.00031499999999999974 | 0 | 0 | 0 | nan | 0 | 0 | 0.008601719547848816 | 0.005739963019926025 | -0.0031767565279227903 | -0.004264944277293292 | -0.006404341160292353 | 0.005879657894338016 | -1.0892370398726083 | 0.4638713229638145 | -0.16271471376900504 | False | 0.008567999999999992 | 0.136 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。