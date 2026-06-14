# 实验总结

## 配置
- model: XGBoost
- horizon: 20
- feature_top_k: 50
- target_label: reduce_position_worth_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.402720696708123
- avg_excess_return_vs_buy_hold: -0.0033071684991918446
- positive_fold_ratio: 0.0
- avg_missed_upside: 0.010741026216852688
- avg_avoided_downside: 0.007372598571132284

## 测试期最终表现
- test_total_return: 1.0962247633280215
- test_excess_return_vs_buy_hold: -0.012515920052099894
- test_max_drawdown: -0.17260282010543815
- test_sharpe: 2.321469982522413

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.2738011335815719 | -0.4928925665966314 | 0.5608436003939219 | -0.9433288296219645 | -0.46104760078474205 | -1.0690708849968777 | 0.5128205128205128 | 0.9865384615384616 | 0.937 | 1.0 | 25 | 0.37799999999999967 | 0.00037799999999999965 | 0 | 0 | 0 | nan | 0 | 0 | 0.023828289695687396 | 0.019185526423288374 | -0.005020763272399021 | -0.002065047654413843 | -0.010813951663628684 | 0.017381307305891602 | -0.6221598567538797 | -0.27173608592715803 | -0.46104760078474205 | False | 0.01346153846153845 | 0.21367521367521367 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.4560148651206528 | 1.0645155056591662 | 0.39426653415907303 | 2.119561164372375 | -0.16271471376900515 | 6.542220313095873 | 0.576 | 0.9929439999999999 | 0.937 | 1.0 | 14 | 0.1259999999999999 | 0.0001259999999999999 | 0 | 0 | 0 | nan | 0 | 0 | 0.008394788954870664 | 0.0029322692901084784 | -0.0055885196647621855 | -0.00785645784316169 | -0.011266455644160569 | 0.0052318116808916975 | -2.1534520604610794 | 0.4638713229638145 | -0.16271471376900504 | False | 0.007055999999999994 | 0.112 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。