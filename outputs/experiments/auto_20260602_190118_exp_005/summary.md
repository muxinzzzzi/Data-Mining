# 实验总结

## 配置
- model: LightGBM
- horizon: 10
- feature_top_k: 50
- target_label: reduce_position_worth_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.3983267415775065
- avg_excess_return_vs_buy_hold: -0.003536468524428608
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.017474785029905692
- avg_avoided_downside: 0.01492388100269266

## 测试期最终表现
- test_total_return: 1.0434774058130016
- test_excess_return_vs_buy_hold: -0.0652632775671198
- test_max_drawdown: -0.17260282010543804
- test_sharpe: 2.26779587775414

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.2687386783644977 | -0.485247803011002 | 0.5600048780008848 | -0.919146764126788 | -0.46104760078474205 | -1.0524895958358078 | 0.5128205128205128 | 0.9842735042735042 | 0.92 | 1.0 | 59 | 0.8319999999999994 | 0.0008319999999999994 | 0 | 0 | 0 | nan | 0 | 0 | 0.02408993969067925 | 0.02655590689902612 | 0.001633967208346871 | 0.0029974075626603147 | 0.0035193139872086643 | 0.01874807082231371 | 0.18771605999162447 | -0.27173608592715803 | -0.46104760078474205 | False | 0.015726495726495728 | 0.5042735042735043 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.45026450982786836 | 1.0492335709546672 | 0.38793991842803843 | 2.1271766508449415 | -0.1611596444575869 | 6.510522994053878 | 0.576 | 0.9783679999999998 | 0.92 | 0.984 | 125 | 0.6559999999999995 | 0.0006559999999999995 | 0 | 0 | 0 | nan | 0 | 0 | 0.02833441539903783 | 0.018215736109051858 | -0.010774679289985972 | -0.013606813135946139 | -0.021721753448611717 | 0.008986380429475218 | -2.417186053838165 | 0.4638713229638145 | -0.16271471376900504 | False | 0.021632000000000012 | 1.0 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。