# 实验总结

## 配置
- model: XGBoost
- horizon: 20
- feature_top_k: 80
- target_label: avoid_loss_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.476156140839002
- avg_excess_return_vs_buy_hold: -0.0046111481401115055
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.015357577078334858
- avg_avoided_downside: 0.012375444051632245

## 测试期最终表现
- test_total_return: 1.0842945051637631
- test_excess_return_vs_buy_hold: -0.024446178216358305
- test_max_drawdown: -0.1700997810573992
- test_sharpe: 2.327829601794959

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.26690616170387194 | -0.4824654276100605 | 0.5631672088465365 | -0.9012909812999913 | -0.46104760078474205 | -1.046454697495147 | 0.5128205128205128 | 0.9883504273504273 | 0.945 | 1.0 | 37 | 0.4760000000000004 | 0.00047600000000000046 | 0 | 0 | 0 | nan | 0 | 0 | 0.014052368340193418 | 0.019481581540038522 | 0.004953213199845104 | 0.004829924223286097 | 0.01066845919966637 | 0.012688305935763694 | 0.840810369302011 | -0.27173608592715803 | -0.46104760078474205 | False | 0.011649572649572662 | 0.3162393162393162 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.4452079543201939 | 1.0337400922413642 | 0.3866438553373784 | 2.114744171453525 | -0.15917400559181683 | 6.494402703493371 | 0.576 | 0.9785920000000004 | 0.945 | 1.0 | 108 | 0.15100000000000013 | 0.00015100000000000015 | 0 | 0 | 0 | nan | 0 | 0 | 0.03202036289481116 | 0.017644750614858214 | -0.014526612279952946 | -0.018663368643620615 | -0.02928565035638515 | 0.009299377674948891 | -3.149205396321975 | 0.4638713229638145 | -0.16271471376900504 | False | 0.021408000000000024 | 0.864 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。