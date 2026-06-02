# 实验总结

## 配置
- model: HistGradientBoosting
- horizon: 10
- feature_top_k: 50
- target_label: reduce_position_worth_label
- strategy_family: ml_ultra_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.421549640003656
- avg_excess_return_vs_buy_hold: -0.0025134980705158925
- positive_fold_ratio: 0.0
- avg_missed_upside: 0.006466021409442444
- avg_avoided_downside: 0.004480353428645867

## 测试期最终表现
- test_total_return: 1.099741988108771
- test_excess_return_vs_buy_hold: -0.008998695271350332
- test_max_drawdown: -0.17260282010543815
- test_sharpe: 2.321935982298348

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_ultra_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_ultra_conservative_full_participation_enhancement | -0.2718649363244984 | -0.4899759704482386 | 0.5631283812993083 | -0.927157728179766 | -0.46104760078474205 | -1.0627448654200955 | 0.5128205128205128 | 0.9934871794871795 | 0.94 | 1.0 | 18 | 0.6520000000000006 | 0.0006520000000000007 | 0 | 0 | 0 | nan | 0 | 0 | 0.014133619593302848 | 0.012992155758767021 | -0.001793463834535827 | -0.00012885039734034098 | -0.003862845182077197 | 0.01376443541157437 | -0.28063956614079283 | -0.27173608592715803 | -0.46104760078474205 | False | 0.006512820512820519 | 0.15384615384615385 | 2024H1 |
| ml_ultra_conservative_full_participation_enhancement | 0.45645967914960717 | 1.0657872175698526 | 0.3946257105091231 | 2.1195578562858377 | -0.16308204174679863 | 6.535282525004164 | 0.576 | 0.9975360000000001 | 0.94 | 1.0 | 10 | 0.39600000000000035 | 0.0003960000000000004 | 0 | 0 | 0 | nan | 0 | 0 | 0.0052644446350244845 | 0.00044890452717058157 | -0.0052115401078539035 | -0.007411643814207336 | -0.010506464857433468 | 0.003399302489473892 | -3.090770794881378 | 0.4638713229638145 | -0.16271471376900504 | False | 0.002464000000000002 | 0.08 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。