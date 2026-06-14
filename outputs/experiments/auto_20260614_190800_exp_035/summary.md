# 实验总结

## 配置
- model: RandomForest
- horizon: 20
- feature_top_k: 80
- target_label: big_down_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.1077551379053279
- avg_excess_return_vs_buy_hold: -0.006284956840771756
- positive_fold_ratio: 0.0
- avg_missed_upside: 0.02042022651663767
- avg_avoided_downside: 0.014860457264085304

## 测试期最终表现
- test_total_return: 1.085171117025559
- test_excess_return_vs_buy_hold: -0.023569566354562532
- test_max_drawdown: -0.17009978105739976
- test_sharpe: 2.3241002669334674

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.27291590439148194 | -0.4915602110003755 | 0.5626976434707139 | -0.9337142104135611 | -0.46104760078474205 | -1.0661810410979222 | 0.5128205128205128 | 0.979717948717949 | 0.934 | 1.0 | 77 | 0.36699999999999944 | 0.00036699999999999943 | 0 | 0 | 0 | nan | 0 | 0 | 0.024708845515390522 | 0.0217549024827985 | -0.0033209430325920226 | -0.0011798184643239118 | -0.0071528003778905105 | 0.010932800592618768 | -0.6542514259996375 | -0.27173608592715803 | -0.46104760078474205 | False | 0.02028205128205129 | 0.6581196581196581 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.44619627090582314 | 1.0411556028487343 | 0.38554993206026517 | 2.123239825771095 | -0.1589233177700472 | 6.55130799846015 | 0.576 | 0.9735600000000002 | 0.934 | 0.977 | 125 | 0.3239999999999995 | 0.00032399999999999947 | 0 | 0 | 0 | nan | 0 | 0 | 0.03655183403452249 | 0.022826469309457417 | -0.01404936472506507 | -0.01767505205799136 | -0.028323519285731222 | 0.010292388319360036 | -2.751889882784012 | 0.4638713229638145 | -0.16271471376900504 | False | 0.02644000000000002 | 1.0 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。