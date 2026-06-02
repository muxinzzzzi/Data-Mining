# 实验总结

## 配置
- model: ExtraTrees
- horizon: 5
- feature_top_k: 50
- target_label: reduce_position_worth_label
- strategy_family: ml_ultra_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.6442102753479744
- avg_excess_return_vs_buy_hold: -0.0013585561792440008
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.0028380584965634428
- avg_avoided_downside: 0.0021311630488074205

## 测试期最终表现
- test_total_return: 1.1001943952226778
- test_excess_return_vs_buy_hold: -0.008546288157443627
- test_max_drawdown: -0.17260282010543793
- test_sharpe: 2.3176398468201365

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_ultra_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_ultra_conservative_full_participation_enhancement | -0.27107056103122595 | -0.4887767683558869 | 0.5682416668088418 | -0.9094706023970256 | -0.46104760078474205 | -1.0601438279343554 | 0.5128205128205128 | 0.9977777777777779 | 0.974 | 1.0 | 10 | 0.26000000000000023 | 0.00026000000000000025 | 0 | 0 | 0 | nan | 0 | 0 | 0.002877267177608586 | 0.003809039613456662 | 0.0006717724358480757 | 0.0006655248959320792 | 0.001446894477211225 | 0.0036433898865364284 | 0.39712864180635593 | -0.27173608592715803 | -0.46104760078474205 | False | 0.0022222222222222244 | 0.08547008547008547 | 2024H1 |
| ml_ultra_conservative_full_participation_enhancement | 0.4591301295301504 | 1.0734302638702138 | 0.39432902359898814 | 2.130248099148245 | -0.16308204174679852 | 6.582148790710038 | 0.576 | 0.996336 | 0.94 | 1.0 | 15 | 0.38000000000000034 | 0.00038000000000000035 | 0 | 0 | 0 | nan | 0 | 0 | 0.005636908312081743 | 0.0025844495329655996 | -0.0034324587791161433 | -0.0047411934336640815 | -0.0069198368986981475 | 0.003427536396912631 | -2.0188952347613935 | 0.4638713229638145 | -0.16271471376900504 | False | 0.003664000000000003 | 0.12 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。