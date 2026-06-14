# 实验总结

## 配置
- model: ExtraTrees
- horizon: 10
- feature_top_k: 50
- target_label: big_down_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.3118155126429285
- avg_excess_return_vs_buy_hold: -0.006438900078773013
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.021239670868820577
- avg_avoided_downside: 0.016426448353816864

## 测试期最终表现
- test_total_return: 1.084207042323027
- test_excess_return_vs_buy_hold: -0.024533641057094435
- test_max_drawdown: -0.1700997810573992
- test_sharpe: 2.325366222149351

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.2695694003137774 | -0.4865064712129593 | 0.5620528072662301 | -0.9179015644082424 | -0.46104760078474205 | -1.0552196137337753 | 0.5128205128205128 | 0.98217094017094 | 0.923 | 1.0 | 70 | 0.46899999999999953 | 0.00046899999999999953 | 0 | 0 | 0 | nan | 0 | 0 | 0.023099790823662248 | 0.024653749664872257 | 0.0010849588412100094 | 0.0021666856133806256 | 0.0023368344272215536 | 0.013658252670103342 | 0.17109321987699563 | -0.27173608592715803 | -0.46104760078474205 | False | 0.017829059829059832 | 0.5982905982905983 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.44238793711411484 | 1.031099850443573 | 0.3850657360498353 | 2.1115892870810313 | -0.16034608445312937 | 6.430464790956421 | 0.576 | 0.9705439999999997 | 0.923 | 1.0 | 122 | 0.7909999999999993 | 0.0007909999999999992 | 0 | 0 | 0 | nan | 0 | 0 | 0.04061922178279949 | 0.024625595396578333 | -0.016784626386221156 | -0.021483385849699665 | -0.03383780679462187 | 0.012243366134743753 | -2.7637666326581747 | 0.4638713229638145 | -0.16271471376900504 | False | 0.02945600000000001 | 0.976 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。