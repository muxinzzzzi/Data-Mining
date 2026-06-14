# 实验总结

## 配置
- model: ExtraTrees
- horizon: 20
- feature_top_k: 50
- target_label: big_down_label
- strategy_family: ml_direct_signal_timing
- search_engine: optuna

## 验证期最优依据
- robust_score: 0.2358546248689506
- avg_excess_return_vs_buy_hold: -0.02718950415314351
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.12461023724260949
- avg_avoided_downside: 0.10303345300509893

## 测试期最终表现
- test_total_return: 0.9531404373001704
- test_excess_return_vs_buy_hold: -0.15560024607995104
- test_max_drawdown: -0.13901030727345653
- test_sharpe: 2.351633551390587

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_direct_signal_timing | 0.20919366220685642 | 0.4504783968692454 | 0.16434268006372446 | 2.4315805700853264 | -0.0688301949638398 | 6.544778742903546 | 0.5403225806451613 | 0.8672177419354842 | 0.8 | 1.0 | 116 | 2.1950000000000003 | 0.002195 | 8 | 53 | 63 | 0.5 | 24 | 29 | 0.07256534294186817 | 0.052281855828122295 | -0.022478487113745876 | -0.025704664785186404 | -0.04568208671503197 | 0.021909010867052364 | -2.0850821149452483 | 0.23489832699204283 | -0.07129994372538018 | False | 0.1327822580645161 | 0.9354838709677419 | 2023H2 |
| ml_direct_signal_timing | -0.2473381343270944 | -0.45262097847889027 | 0.52075729848993 | -0.9109390401792877 | -0.43073200242461307 | -1.0508180862602803 | 0.5128205128205128 | 0.9149999999999999 | 0.915 | 0.915 | 117 | 0.08499999999999996 | 8.499999999999997e-05 | 0 | 117 | 0 | nan | 57 | 60 | 0.11790331648268913 | 0.1383555241246219 | 0.020367207641932773 | 0.02439795160006364 | 0.04386783184416284 | 0.04837559894985392 | 0.9068173375927847 | -0.27173608592715803 | -0.46104760078474205 | False | 0.085 | 1.0 | 2024H1 |
| ml_direct_signal_timing | 0.38360952368950674 | 0.8682037617469103 | 0.34155826254514055 | 2.0886937206265146 | -0.13751079880892092 | 6.313713317550637 | 0.568 | 0.8680000000000003 | 0.8 | 0.935 | 125 | 1.335 | 0.0013349999999999998 | 0 | 72 | 53 | nan | 32 | 40 | 0.18336205230327113 | 0.11846297906255261 | -0.06623407324071852 | -0.08026179927430777 | -0.13352789165328854 | 0.059252900540811476 | -2.2535249824828214 | 0.4638713229638145 | -0.16271471376900504 | False | 0.132 | 1.0 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。