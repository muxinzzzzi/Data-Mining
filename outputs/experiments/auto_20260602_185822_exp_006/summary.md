# 实验总结

## 配置
- model: LightGBM
- horizon: 10
- feature_top_k: 50
- target_label: reduce_position_worth_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.2563108208823934
- avg_excess_return_vs_buy_hold: -0.005965820345317045
- positive_fold_ratio: 0.0
- avg_missed_upside: 0.019938022452106588
- avg_avoided_downside: 0.013886888084266356

## 测试期最终表现
- test_total_return: 1.0738627348580838
- test_excess_return_vs_buy_hold: -0.03487794852203763
- test_max_drawdown: -0.17260282010543793
- test_sharpe: 2.2929351406023435

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.27509370042566583 | -0.4948346413761091 | 0.5617514218418049 | -0.9477070388737676 | -0.46104760078474205 | -1.0732831936092035 | 0.5128205128205128 | 0.9899145299145299 | 0.941 | 1.0 | 20 | 0.5900000000000005 | 0.0005900000000000005 | 0 | 0 | 0 | nan | 0 | 0 | 0.02062710697506947 | 0.014656846830836165 | -0.006560260144233304 | -0.0033576144985077994 | -0.014129791079887105 | 0.015830440050909222 | -0.892570960405839 | -0.27173608592715803 | -0.46104760078474205 | False | 0.010085470085470094 | 0.17094017094017094 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.44933147642637117 | 1.0474882320637353 | 0.38378851452424184 | 2.1426352757660796 | -0.1582487963793624 | 6.6192492835941765 | 0.576 | 0.9710000000000001 | 0.971 | 0.971 | 125 | 0.029000000000000026 | 2.9000000000000027e-05 | 0 | 0 | 0 | nan | 0 | 0 | 0.039186960381250295 | 0.0270038174219629 | -0.012212142959287393 | -0.014539846537443335 | -0.02461968020592341 | 0.011464327787338883 | -2.1475031648269165 | 0.4638713229638145 | -0.16271471376900504 | False | 0.029000000000000022 | 1.0 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。