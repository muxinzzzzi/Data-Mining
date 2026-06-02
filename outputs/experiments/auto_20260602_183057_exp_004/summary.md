# 实验总结

## 配置
- model: HistGradientBoosting
- horizon: 3
- feature_top_k: 80
- target_label: reduce_position_worth_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.3165216496693677
- avg_excess_return_vs_buy_hold: -0.007913979374740551
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.026849534527492835
- avg_avoided_downside: 0.021128924331918014

## 测试期最终表现
- test_total_return: 1.076055899635882
- test_excess_return_vs_buy_hold: -0.03268478374423944
- test_max_drawdown: -0.16826696450048995
- test_sharpe: 2.3302166651791922

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.2316022279494534 | 0.49958212357649456 | 0.17762458954733784 | 2.472855336686021 | -0.07039174188857811 | 7.0971694999006365 | 0.5403225806451613 | 0.9864354838709678 | 0.971 | 1.0 | 58 | 1.044000000000001 | 0.0010440000000000009 | 0 | 0 | 0 | nan | 0 | 0 | 0.00935699314631284 | 0.007421725743884459 | -0.0029792674024283813 | -0.0032960990425894288 | -0.0060546402049350915 | 0.004324142954367386 | -1.4001942740630031 | 0.23489832699204283 | -0.07129994372538018 | False | 0.01356451612903227 | 0.46774193548387094 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.27052548429444 | -0.48795303862152417 | 0.5568923729863398 | -0.9366905543410937 | -0.46077388252817697 | -1.0589858868393764 | 0.5128205128205128 | 0.9754615384615385 | 0.92 | 1.0 | 48 | 1.3159999999999994 | 0.0013159999999999995 | 0 | 0 | 0 | nan | 0 | 0 | 0.03338127252799165 | 0.03312341821258015 | -0.0015738543154114968 | 0.0012106016327180225 | -0.0033898400639632087 | 0.02278044642842743 | -0.1488048126981862 | -0.27173608592715803 | -0.46104760078474205 | False | 0.024538461538461533 | 0.41025641025641024 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.44221488224946426 | 1.0308175094026302 | 0.3849073397446096 | 2.111735019963692 | -0.15943478176860615 | 6.465449370380771 | 0.576 | 0.9741600000000002 | 0.92 | 1.0 | 85 | 1.954 | 0.001954 | 0 | 0 | 0 | nan | 0 | 0 | 0.03781033790817403 | 0.02284162903928943 | -0.016922708868884598 | -0.021656440714350245 | -0.03411618107967139 | 0.0133389627158354 | -2.557633738579255 | 0.4638713229638145 | -0.16271471376900504 | False | 0.02584000000000001 | 0.68 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。