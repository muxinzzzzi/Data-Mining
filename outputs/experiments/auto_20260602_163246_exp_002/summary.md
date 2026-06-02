# 实验总结

## 配置
- model: LightGBM
- horizon: 5
- feature_top_k: 50
- target_label: big_down_label
- strategy_family: ml_big_up_index_enhancement

## 验证结果
- robust_score: 1.3502444147973303
- avg_excess_return_vs_buy_hold: -0.009543441212799856
- positive_fold_ratio: 0.3333333333333333

## 测试结果
- test_total_return: 1.0012962861205552
- test_excess_return_vs_buy_hold: -0.10744439725956623
- test_max_drawdown: -0.16592212280874363
- test_sharpe: 2.299609499523079

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_big_up_index_enhancement | 0.20888657494737028 | 0.44506986479790256 | 0.1694919320412995 | 2.359795550608187 | -0.06684696339502805 | 6.658041625133955 | 0.5403225806451613 | 0.9404838709677423 | 0.9 | 1.0 | 111 | 0.7399999999999997 | 0.0007399999999999998 | 0 | 0 | 0 | nan | 0 | 0 | 0.04394060911383607 | 0.022376233458925275 | -0.022304375654910797 | -0.026011752044672543 | -0.04532824729868967 | 0.012867741376761558 | -3.5226265411698456 | 0.23489832699204283 | -0.07129994372538018 | False | 0.05951612903225806 | 0.8951612903225806 | 2023H2 |
| ml_big_up_index_enhancement | -0.24303960382131917 | -0.4459312880054931 | 0.5122204625524976 | -0.9110076728236985 | -0.4252333635316178 | -1.0486742721736986 | 0.5128205128205128 | 0.9000000000000005 | 0.9 | 0.9 | 117 | 0.09999999999999998 | 9.999999999999998e-05 | 0 | 0 | 0 | nan | 0 | 0 | 0.13870978409728138 | 0.16277120485249638 | 0.023961420755215002 | 0.02869648210583886 | 0.0516092139343092 | 0.056912469352769356 | 0.9068173375927836 | -0.27173608592715803 | -0.46104760078474205 | False | 0.09999999999999998 | 1.0 | 2024H1 |
| ml_big_up_index_enhancement | 0.4325562692642486 | 1.0048668846491586 | 0.36605171596239866 | 2.1639240350888924 | -0.15408568483007945 | 6.52148112108722 | 0.576 | 0.9177600000000006 | 0.9 | 0.96 | 125 | 0.33999999999999975 | 0.00033999999999999975 | 0 | 0 | 0 | nan | 0 | 0 | 0.1036378558641104 | 0.07678024494999369 | -0.027197610914116703 | -0.03131505369956589 | -0.054830383602859314 | 0.03169415683195114 | -1.7299839807564892 | 0.4638713229638145 | -0.16271471376900504 | False | 0.08223999999999998 | 1.0 | 2024H2 |

## 诚实分析
- 当前策略未稳定跑赢买入持有，主要原因通常是测试期单边上涨较强，减仓触发仍会错失上行收益。
- 即使多折验证更稳健，验证期学到的防守模式在 2025-01-01 到 2026-05-06 的强趋势环境中可能仍然偏保守。