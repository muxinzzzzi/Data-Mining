# 实验总结

## 配置
- model: ExtraTrees
- horizon: 10
- feature_top_k: 50
- target_label: avoid_loss_label
- strategy_family: ml_big_up_index_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 0.4911327252963349
- avg_excess_return_vs_buy_hold: -0.012420464993399364
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.15171831140384195
- avg_avoided_downside: 0.14146112856841864

## 测试期最终表现
- test_total_return: 0.9487596485106731
- test_excess_return_vs_buy_hold: -0.15998103486944837
- test_max_drawdown: -0.16788913280294482
- test_sharpe: 2.2789556840197145

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_big_up_index_enhancement | 0.1942549584467701 | 0.40861409148759487 | 0.16322001166953495 | 2.29239455263653 | -0.0640942419475703 | 6.3752074924584505 | 0.5403225806451613 | 0.9212096774193548 | 0.83 | 1.0 | 69 | 1.1219999999999999 | 0.0011220000000000002 | 0 | 0 | 0 | nan | 0 | 0 | 0.06512476764726552 | 0.031246346193331533 | -0.035000421453933984 | -0.04064336854527273 | -0.07112988876122073 | 0.021440614713433472 | -3.3175302906149784 | 0.23489832699204283 | -0.07129994372538018 | False | 0.07879032258064518 | 0.5564516129032258 | 2023H2 |
| ml_big_up_index_enhancement | -0.22302187152968433 | -0.4142146744833579 | 0.4723819113156085 | -0.9113607263445357 | -0.3989671955687324 | -1.0382173749721204 | 0.5128205128205128 | 0.8299999999999998 | 0.83 | 0.83 | 117 | 0.17000000000000004 | 0.00017000000000000004 | 0 | 0 | 0 | nan | 0 | 0 | 0.2358066329653785 | 0.2767110482492439 | 0.04073441528386543 | 0.0487142143974737 | 0.08773566368832565 | 0.09675119789970792 | 0.9068173375927836 | -0.27173608592715803 | -0.46104760078474205 | False | 0.17000000000000007 | 1.0 | 2024H1 |
| ml_big_up_index_enhancement | 0.41853908213141544 | 0.970243505063014 | 0.3554033091083556 | 2.1617787364928653 | -0.1555090465141753 | 6.239145096774626 | 0.576 | 0.8723119999999996 | 0.83 | 0.99 | 125 | 1.2080000000000002 | 0.001208 | 0 | 0 | 0 | nan | 0 | 0 | 0.15422353359888183 | 0.11642599126268045 | -0.039005542336201376 | -0.04533224083239906 | -0.078635173349782 | 0.05147661868320576 | -1.5275901052031748 | 0.4638713229638145 | -0.16271471376900504 | False | 0.12768800000000005 | 1.0 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。