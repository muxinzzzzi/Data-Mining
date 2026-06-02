# 实验总结

## 配置
- model: RandomForest
- horizon: 10
- feature_top_k: 80
- target_label: big_up_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.3640546772511735
- avg_excess_return_vs_buy_hold: -0.0035597447033316096
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.021361967098581902
- avg_avoided_downside: 0.020010315663230485

## 测试期最终表现
- test_total_return: 1.0492896262183287
- test_excess_return_vs_buy_hold: -0.059451057161792775
- test_max_drawdown: -0.168095429194564
- test_sharpe: 2.3093961450289195

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.2615201481396465 | -0.4742411164421645 | 0.5603829602540614 | -0.8803687817109876 | -0.45794314524383095 | -1.0355895079282291 | 0.5128205128205128 | 0.9786153846153849 | 0.92 | 1.0 | 72 | 0.45299999999999974 | 0.0004529999999999998 | 0 | 0 | 0 | nan | 0 | 0 | 0.022457018219949783 | 0.034471810338669764 | 0.011561792118719981 | 0.010215937787511509 | 0.024902321486473783 | 0.015078798813879796 | 1.6514791260131139 | -0.27173608592715803 | -0.46104760078474205 | False | 0.021384615384615398 | 0.6153846153846154 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.44297615106630817 | 1.0292893514659047 | 0.38402625284683883 | 2.1184119862543347 | -0.15900113154992068 | 6.473471864209624 | 0.576 | 0.9704560000000002 | 0.92 | 0.973 | 125 | 0.5039999999999994 | 0.0005039999999999995 | 0 | 0 | 0 | nan | 0 | 0 | 0.04162888307579593 | 0.02555913665102169 | -0.01657374642477424 | -0.020895171897506337 | -0.03341267279234486 | 0.011815819609547112 | -2.8277913760081117 | 0.4638713229638145 | -0.16271471376900504 | False | 0.029544000000000025 | 1.0 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。