# 实验总结

## 配置
- model: HistGradientBoosting
- horizon: 10
- feature_top_k: 50
- target_label: reduce_position_worth_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.4150051877009164
- avg_excess_return_vs_buy_hold: -0.0072737716521726154
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.01982454530012269
- avg_avoided_downside: 0.0140482352857556

## 测试期最终表现
- test_total_return: 1.053364524988917
- test_excess_return_vs_buy_hold: -0.055376158391204466
- test_max_drawdown: -0.17260282010543815
- test_sharpe: 2.291769773915523

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.27068225231566734 | -0.4881900220489477 | 0.5586669681318852 | -0.9327701390830405 | -0.46104760078474205 | -1.0588711907794486 | 0.5128205128205128 | 0.9882051282051283 | 0.92 | 1.0 | 27 | 0.8399999999999994 | 0.0008399999999999994 | 0 | 0 | 0 | nan | 0 | 0 | 0.024023686074137397 | 0.023534956042314998 | -0.001328730031822398 | 0.0010538336114906954 | -0.002861880068540547 | 0.02134453774506681 | -0.13408020837565293 | -0.27173608592715803 | -0.46104760078474205 | False | 0.011794871794871792 | 0.23076923076923078 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.44099617439580596 | 1.023194503112078 | 0.3863905767467127 | 2.100615634756402 | -0.16048739564148673 | 6.375544316251447 | 0.576 | 0.9769600000000004 | 0.92 | 1.0 | 117 | 0.6599999999999995 | 0.0006599999999999995 | 0 | 0 | 0 | nan | 0 | 0 | 0.035449949826230676 | 0.018609749814951806 | -0.01750020001127887 | -0.02287514856800854 | -0.03528040322273819 | 0.010393688338707914 | -3.394406496810934 | 0.4638713229638145 | -0.16271471376900504 | False | 0.023040000000000012 | 0.936 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。