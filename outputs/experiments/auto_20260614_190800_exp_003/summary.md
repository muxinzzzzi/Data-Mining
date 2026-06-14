# 实验总结

## 配置
- model: RandomForest
- horizon: 20
- feature_top_k: 80
- target_label: big_down_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.5589046393200339
- avg_excess_return_vs_buy_hold: -0.0012755855214399148
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.014107022357681418
- avg_avoided_downside: 0.013160544627511393

## 测试期最终表现
- test_total_return: 1.084761288463155
- test_excess_return_vs_buy_hold: -0.023979394916966434
- test_max_drawdown: -0.17009978105739976
- test_sharpe: 2.32811249691446

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23315652451340418 | 0.5034306575567022 | 0.17895438001302685 | 2.470139344317194 | -0.07045438697418482 | 7.145483470619993 | 0.5403225806451613 | 0.9954274193548388 | 0.973 | 1.0 | 21 | 0.27000000000000024 | 0.0002700000000000003 | 0 | 0 | 0 | nan | 0 | 0 | 0.004495150556781849 | 0.003164812714628982 | -0.0016003378421528668 | -0.0017418024786386432 | -0.0032522994856654867 | 0.003284095102345705 | -0.9903183020925588 | 0.23489832699204283 | -0.07129994372538018 | False | 0.004572580645161295 | 0.1693548387096774 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.26955329644224535 | -0.48648208712498586 | 0.5628384181094656 | -0.9157816848893037 | -0.45949642652575184 | -1.0587287714145512 | 0.5128205128205128 | 0.9855726495726496 | 0.945 | 1.0 | 48 | 0.5210000000000005 | 0.0005210000000000004 | 0 | 0 | 0 | nan | 0 | 0 | 0.018094657588737996 | 0.01991977610026956 | 0.0013041185115315638 | 0.002182789484912684 | 0.0028088706402217872 | 0.010818365628257404 | 0.259639093070127 | -0.27173608592715803 | -0.46104760078474205 | False | 0.014427350427350443 | 0.41025641025641024 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.4596035793932207 | 1.078700100794618 | 0.3906875115968472 | 2.1480666819570353 | -0.15805449077053302 | 6.824862081019251 | 0.576 | 0.9819920000000001 | 0.945 | 1.0 | 73 | 0.49300000000000044 | 0.0004930000000000004 | 0 | 0 | 0 | nan | 0 | 0 | 0.019731258927524406 | 0.01639704506763564 | -0.0038272138598887673 | -0.004267743570593785 | -0.0077156631415357495 | 0.0077282698337843994 | -0.9983687562003154 | 0.4638713229638145 | -0.16271471376900504 | False | 0.01800800000000002 | 0.584 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。