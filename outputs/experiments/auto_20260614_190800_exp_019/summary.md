# 实验总结

## 配置
- model: XGBoost
- horizon: 20
- feature_top_k: 50
- target_label: reduce_position_worth_label
- strategy_family: ml_big_up_index_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 0.8456661680456475
- avg_excess_return_vs_buy_hold: -0.004987950168939825
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.08189043490639064
- avg_avoided_downside: 0.07790654075342301

## 测试期最终表现
- test_total_return: 1.0310697658852064
- test_excess_return_vs_buy_hold: -0.07767091749491506
- test_max_drawdown: -0.17009978105739965
- test_sharpe: 2.324486763289241

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_big_up_index_enhancement | 0.215774503426029 | 0.4606769567118474 | 0.17221915363737322 | 2.392252817809143 | -0.06724199671422992 | 6.851030296879298 | 0.5403225806451613 | 0.9650161290322583 | 0.903 | 1.0 | 58 | 0.3929999999999998 | 0.0003929999999999998 | 0 | 0 | 0 | nan | 0 | 0 | 0.029939851280043308 | 0.013945759795207235 | -0.016387091484836074 | -0.01912382356601383 | -0.03330279882402172 | 0.01116971626730748 | -2.9815259427399528 | 0.23489832699204283 | -0.07129994372538018 | False | 0.034983870967741915 | 0.46774193548387094 | 2023H2 |
| ml_big_up_index_enhancement | -0.24389909283137157 | -0.4472723081461214 | 0.5139278296483831 | -0.9109937640485036 | -0.4263367163416446 | -1.0491057678168636 | 0.5128205128205128 | 0.9030000000000006 | 0.903 | 0.903 | 117 | 0.09699999999999998 | 9.699999999999997e-05 | 0 | 0 | 0 | nan | 0 | 0 | 0.13454849057436297 | 0.15788806870692146 | 0.023242578132558492 | 0.027836993095786466 | 0.05006093751627992 | 0.055205095272186255 | 0.9068173375927839 | -0.27173608592715803 | -0.46104760078474205 | False | 0.09699999999999996 | 1.0 | 2024H1 |
| ml_big_up_index_enhancement | 0.4401943029272224 | 1.0262666621654848 | 0.37343768621239365 | 2.156995211464377 | -0.15799060202246384 | 6.495744993867201 | 0.576 | 0.9335680000000001 | 0.903 | 1.0 | 119 | 1.2559999999999993 | 0.0012559999999999993 | 0 | 0 | 0 | nan | 0 | 0 | 0.08118296286476565 | 0.061885793758140364 | -0.02055316910662528 | -0.02367702003659211 | -0.04143518891895656 | 0.028330346898101835 | -1.4625725928450515 | 0.4638713229638145 | -0.16271471376900504 | False | 0.06643199999999999 | 0.952 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。