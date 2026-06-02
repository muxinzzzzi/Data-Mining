# 实验总结

## 配置
- model: ExtraTrees
- horizon: 10
- feature_top_k: 80
- target_label: big_up_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.4167518003024817
- avg_excess_return_vs_buy_hold: -0.0018403100089323665
- positive_fold_ratio: 0.0
- avg_missed_upside: 0.004954673151652335
- avg_avoided_downside: 0.0034142757659953085

## 测试期最终表现
- test_total_return: 1.0317959474791079
- test_excess_return_vs_buy_hold: -0.07694473590101358
- test_max_drawdown: -0.1637468155530165
- test_sharpe: 2.325326966460057

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.27173608592715803 | -0.4897815582681906 | 0.5691327216655068 | -0.9105889817411068 | -0.46104760078474205 | -1.0623231905654447 | 0.5128205128205128 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | -0.27173608592715803 | -0.46104760078474205 | True | 0.0 | 0.0 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.4583503929370174 | 1.0719775491999801 | 0.39090425649589783 | 2.1427241493333864 | -0.16102250434618814 | 6.657315097368604 | 0.576 | 0.9890000000000004 | 0.989 | 0.989 | 125 | 0.01100000000000001 | 1.100000000000001e-05 | 0 | 0 | 0 | nan | 0 | 0 | 0.014864019454957007 | 0.010242827297985925 | -0.004632192156971081 | -0.0055209300267970995 | -0.009338499388453684 | 0.004348538126231992 | -2.1475031648269103 | 0.4638713229638145 | -0.16271471376900504 | False | 0.011000000000000015 | 1.0 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。