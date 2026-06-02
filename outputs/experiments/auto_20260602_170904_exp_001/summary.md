# 实验总结

## 配置
- model: ExtraTrees
- horizon: 3
- feature_top_k: 50
- target_label: avoid_loss_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.2967394368317093
- avg_excess_return_vs_buy_hold: -0.005260655013445414
- positive_fold_ratio: 0.0
- avg_missed_upside: 0.021077027533828036
- avg_avoided_downside: 0.015807141142805327

## 测试期最终表现
- test_total_return: 1.0678057861627912
- test_excess_return_vs_buy_hold: -0.040934897217330235
- test_max_drawdown: -0.16792832284837667
- test_sharpe: 2.325441337484228

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.27347875131914867 | -0.4924075669244541 | 0.5593373822056867 | -0.9456760141260218 | -0.46104760078474205 | -1.0680189335902295 | 0.5128205128205128 | 0.9874957264957266 | 0.923 | 1.0 | 19 | 0.9239999999999995 | 0.0009239999999999995 | 0 | 0 | 0 | nan | 0 | 0 | 0.025395396716138986 | 0.0213487721244518 | -0.0049706245916871835 | -0.0017426653919906387 | -0.010705960659018531 | 0.020952923003377005 | -0.5109530855095034 | -0.27173608592715803 | -0.46104760078474205 | False | 0.012504273504273498 | 0.1623931623931624 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.4498320233154689 | 1.048843856144277 | 0.3841838334763214 | 2.1426402998139094 | -0.15840304236008174 | 6.6213618155139065 | 0.576 | 0.9719999999999994 | 0.972 | 0.972 | 125 | 0.028000000000000025 | 2.8000000000000027e-05 | 0 | 0 | 0 | nan | 0 | 0 | 0.03783568588534512 | 0.02607265130396418 | -0.011791034581380939 | -0.014039299648345605 | -0.023770725716063992 | 0.011069006139499614 | -2.1475031648269165 | 0.4638713229638145 | -0.16271471376900504 | False | 0.028000000000000025 | 1.0 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。