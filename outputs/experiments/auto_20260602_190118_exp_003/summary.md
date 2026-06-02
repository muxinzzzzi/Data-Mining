# 实验总结

## 配置
- model: ExtraTrees
- horizon: 5
- feature_top_k: 50
- target_label: reduce_position_worth_label
- strategy_family: ml_big_up_plus_115
- search_engine: optuna

## 验证期最优依据
- robust_score: 0.6329875400966691
- avg_excess_return_vs_buy_hold: -0.007488620921370925
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.12347087055874091
- avg_avoided_downside: 0.11718857351860823

## 测试期最终表现
- test_total_return: 0.9985814304513749
- test_excess_return_vs_buy_hold: -0.11015925292874651
- test_max_drawdown: -0.16722368701403867
- test_sharpe: 2.3057526331056795

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_big_up_plus_115 | 0.20131892350875913 | 0.4255983380948536 | 0.17388052445668017 | 2.231037850941114 | -0.06898253593886972 | 6.169653410132765 | 0.532258064516129 | 0.9532096774193548 | 0.856 | 1.15 | 71 | 1.8779999999999997 | 0.0018779999999999997 | 0 | 0 | 0 | nan | 0 | 0 | 0.045264015004273205 | 0.01808426450915969 | -0.029057750495113516 | -0.0335794034832837 | -0.057360522789387906 | 0.016416475489122496 | -3.4940826870782833 | 0.23489832699204283 | -0.07129994372538018 | False | 0.06856451612903226 | 0.5725806451612904 | 2023H2 |
| ml_big_up_plus_115 | -0.23044797380674686 | -0.4260890952437788 | 0.48717908410415034 | -0.9112228581156188 | -0.40884073558043077 | -1.0421884566831643 | 0.5128205128205128 | 0.8559999999999998 | 0.856 | 0.856 | 117 | 0.14400000000000002 | 0.00014400000000000003 | 0 | 0 | 0 | nan | 0 | 0 | 0.19974208910008526 | 0.23439053498759485 | 0.034504445887509595 | 0.04128811212041117 | 0.07431726806540533 | 0.08195395586798786 | 0.9068173375927847 | -0.27173608592715803 | -0.46104760078474205 | False | 0.144 | 1.0 | 2024H1 |
| ml_big_up_plus_115 | 0.43369675156257426 | 1.0111202417418625 | 0.36211359980265134 | 2.18765804952193 | -0.15567766312636977 | 6.494960300894907 | 0.576 | 0.8937279999999996 | 0.856 | 1.0 | 121 | 0.8460000000000002 | 0.0008460000000000002 | 0 | 0 | 0 | nan | 0 | 0 | 0.1254065075718643 | 0.09909092105907012 | -0.027161586512794184 | -0.030174571401240247 | -0.054757758409793106 | 0.04339037929819446 | -1.261979251978368 | 0.4638713229638145 | -0.16271471376900504 | False | 0.10627200000000006 | 0.968 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。