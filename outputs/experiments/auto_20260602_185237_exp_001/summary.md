# 实验总结

## 配置
- model: LightGBM
- horizon: 10
- feature_top_k: 50
- target_label: reduce_position_worth_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.3670175709848142
- avg_excess_return_vs_buy_hold: -0.007618914764097262
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.024029856099252577
- avg_avoided_downside: 0.018932991598304087

## 测试期最终表现
- test_total_return: 1.050627539147539
- test_excess_return_vs_buy_hold: -0.0581131442325824
- test_max_drawdown: -0.16759407461569598
- test_sharpe: 2.296469331541351

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23269978373225686 | 0.502299219420191 | 0.1793088442123449 | 2.461368547446106 | -0.07036041876373667 | 7.138945848330751 | 0.5403225806451613 | 0.992983870967742 | 0.97 | 1.0 | 29 | 0.6600000000000006 | 0.0006600000000000006 | 0 | 0 | 0 | nan | 0 | 0 | 0.004832166713914991 | 0.0035488087125312637 | -0.001943358001383728 | -0.0021985432597859678 | -0.00394940497055404 | 0.003153540646657631 | -1.2523716714227002 | 0.23489832699204283 | -0.07129994372538018 | False | 0.007016129032258071 | 0.23387096774193547 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.2672361670556469 | -0.4829670797387122 | 0.5595041578713712 | -0.9125860974235489 | -0.46097152725356216 | -1.0477156422571219 | 0.5128205128205128 | 0.9829059829059829 | 0.92 | 1.0 | 45 | 0.9499999999999997 | 0.0009499999999999999 | 0 | 0 | 0 | nan | 0 | 0 | 0.023331406385989403 | 0.027833317259100456 | 0.0035519108731110536 | 0.004499918871511133 | 0.007650269572854612 | 0.01936914068111366 | 0.3949720691695006 | -0.27173608592715803 | -0.46104760078474205 | False | 0.017094017094017096 | 0.38461538461538464 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.43871320305979755 | 1.0174287609419546 | 0.3831293304548429 | 2.106873023294748 | -0.1594793319477893 | 6.379690386933918 | 0.576 | 0.9712000000000002 | 0.92 | 1.0 | 100 | 1.1999999999999995 | 0.0011999999999999997 | 0 | 0 | 0 | nan | 0 | 0 | 0.04392599519785334 | 0.025416848823280547 | -0.019709146374572793 | -0.02515811990401695 | -0.039733639091138744 | 0.013300710794068247 | -2.987332008516331 | 0.4638713229638145 | -0.16271471376900504 | False | 0.02880000000000002 | 0.8 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。