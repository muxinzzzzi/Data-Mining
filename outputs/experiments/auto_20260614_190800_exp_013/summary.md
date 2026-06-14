# 实验总结

## 配置
- model: RandomForest
- horizon: 5
- feature_top_k: 80
- target_label: big_up_label
- strategy_family: ml_direct_signal_timing
- search_engine: optuna

## 验证期最优依据
- robust_score: 0.33002559913644147
- avg_excess_return_vs_buy_hold: -0.023497644485817853
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.13792057140275474
- avg_avoided_downside: 0.11965014144990227

## 测试期最终表现
- test_total_return: 0.9041726461624384
- test_excess_return_vs_buy_hold: -0.20456803721768302
- test_max_drawdown: -0.13901030727345598
- test_sharpe: 2.3023011324284375

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_direct_signal_timing | 0.22244725579065028 | 0.47701550991364483 | 0.15617553845629406 | 2.6926042758306528 | -0.05735132298935841 | 8.317428178634266 | 0.532258064516129 | 0.842967741935484 | 0.8 | 1.0 | 120 | 2.077 | 0.0020769999999999994 | 4 | 44 | 76 | 1.0 | 18 | 26 | 0.08311591878477269 | 0.07300173489983602 | -0.012191183884936663 | -0.01245107120139255 | -0.02477563176616159 | 0.028646338809208548 | -0.864879520247708 | 0.23489832699204283 | -0.07129994372538018 | False | 0.15703225806451612 | 0.967741935483871 | 2023H2 |
| ml_direct_signal_timing | -0.24332608769953046 | -0.44637846523514724 | 0.5127895849126247 | -0.9110030262833271 | -0.42560134962244134 | -1.0488182559363066 | 0.5128205128205128 | 0.9009999999999998 | 0.901 | 0.901 | 117 | 0.09899999999999998 | 9.899999999999998e-05 | 0 | 117 | 0 | nan | 57 | 60 | 0.13732268625630853 | 0.1611434928039714 | 0.02372180654766286 | 0.028409998227627575 | 0.051093121794966154 | 0.056343344659241644 | 0.9068173375927847 | -0.27173608592715803 | -0.46104760078474205 | False | 0.09899999999999999 | 1.0 | 2024H1 |
| ml_direct_signal_timing | 0.3774194624801259 | 0.8522814115751711 | 0.33913475503255386 | 2.0744509366165875 | -0.13994082029608745 | 6.0902988118256705 | 0.568 | 0.8592320000000001 | 0.8 | 1.0 | 121 | 2.623 | 0.002623 | 4 | 64 | 57 | 0.25 | 27 | 37 | 0.19332310916718307 | 0.12480519664589941 | -0.07114091252128366 | -0.08645186048368858 | -0.1434200796429079 | 0.06089536813106005 | -2.355188646437554 | 0.4638713229638145 | -0.16271471376900504 | False | 0.14076799999999998 | 0.968 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。