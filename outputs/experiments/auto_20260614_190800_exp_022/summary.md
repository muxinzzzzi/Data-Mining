# 实验总结

## 配置
- model: RandomForest
- horizon: 20
- feature_top_k: 80
- target_label: big_down_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.1045354521510788
- avg_excess_return_vs_buy_hold: -0.006280131901676385
- positive_fold_ratio: 0.0
- avg_missed_upside: 0.021399437976037595
- avg_avoided_downside: 0.015808715297645257

## 测试期最终表现
- test_total_return: 1.082622810674252
- test_excess_return_vs_buy_hold: -0.026117872705869516
- test_max_drawdown: -0.17009978105739954
- test_sharpe: 2.3282377348898073

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.2726161790916899 | -0.4911086709854474 | 0.5623277545610805 | -0.9331297977446994 | -0.46104760078474205 | -1.0652016627991099 | 0.5128205128205128 | 0.9789999999999995 | 0.937 | 1.0 | 77 | 0.3289999999999994 | 0.00032899999999999943 | 0 | 0 | 0 | nan | 0 | 0 | 0.025580270509768695 | 0.022901256974863095 | -0.0030080135349055993 | -0.0008800931645318588 | -0.006478798382873601 | 0.01126071229721824 | -0.5753453433380119 | -0.27173608592715803 | -0.46104760078474205 | False | 0.021 | 0.6581196581196581 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.4459110204233172 | 1.040134207053311 | 0.3848287865825958 | 2.125464785652995 | -0.15864405783761393 | 6.556401930401827 | 0.576 | 0.9719599999999996 | 0.937 | 0.975 | 125 | 0.2909999999999995 | 0.0002909999999999995 | 0 | 0 | 0 | nan | 0 | 0 | 0.0386180434183441 | 0.02452488891807267 | -0.014384154500271428 | -0.017960302540497297 | -0.02899845547254717 | 0.010857948540797687 | -2.670712185049348 | 0.4638713229638145 | -0.16271471376900504 | False | 0.028040000000000013 | 1.0 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。