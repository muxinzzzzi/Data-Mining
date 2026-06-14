# 实验总结

## 配置
- model: RandomForest
- horizon: 20
- feature_top_k: 80
- target_label: big_down_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.6705853246812923
- avg_excess_return_vs_buy_hold: 0.0012050914516164024
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.011342236937359826
- avg_avoided_downside: 0.012225087348409549

## 测试期最终表现
- test_total_return: 1.084449136566143
- test_excess_return_vs_buy_hold: -0.024291546813978382
- test_max_drawdown: -0.17009978105739965
- test_sharpe: 2.3215956343301305

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.267702842227991 | -0.4836760433604963 | 0.5601438029149183 | -0.913250373056074 | -0.46104760078474205 | -1.049080490902108 | 0.5128205128205128 | 0.9857777777777779 | 0.948 | 1.0 | 32 | 0.20800000000000018 | 0.00020800000000000018 | 0 | 0 | 0 | nan | 0 | 0 | 0.021194286887886483 | 0.02451042370066995 | 0.003108136812783466 | 0.004033243699167022 | 0.006694448519841325 | 0.016476229323911667 | 0.4063095012962576 | -0.27173608592715803 | -0.46104760078474205 | False | 0.014222222222222235 | 0.27350427350427353 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.4634533536194967 | 1.0895530998697658 | 0.39250790165702304 | 2.153552073123618 | -0.1617219050696741 | 6.7372017377630895 | 0.576 | 0.986336 | 0.948 | 1.0 | 33 | 0.15200000000000014 | 0.00015200000000000012 | 0 | 0 | 0 | nan | 0 | 0 | 0.01283242392419299 | 0.012164838344558698 | -0.0008195855796342921 | -0.0004179693443178145 | -0.001652284528542726 | 0.007608113667764771 | -0.2171740066849132 | 0.4638713229638145 | -0.16271471376900504 | False | 0.013664000000000013 | 0.264 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。