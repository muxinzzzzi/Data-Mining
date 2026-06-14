# Auto Improve Report

## 搜索设置
- search_engine: optuna
- validation_folds: 2023H2, 2024H1, 2024H2
- test_period: 2025-01-01 to 2026-05-06
- max_runs: 50
- feature_top_k_candidates: [50, 80]
- families: ['ml_big_up_index_enhancement', 'ml_bull_full_participation_enhancement', 'ml_conservative_full_participation_enhancement', 'ml_ultra_conservative_full_participation_enhancement', 'ml_direct_signal_timing']
- target_labels: ['big_up_label', 'big_down_label', 'avoid_loss_label', 'reduce_position_worth_label']

## 可选模型依赖
- skipped_optional_models: []

## 验证期信号预筛选
- signal_leaderboard_path: C:\Users\qqrtq\Documents\GitHub\Data-Mining\outputs\signal_leaderboard.csv
- candidate_count: 16
- top_candidate: ExtraTrees / 20D / top_k=50

## 验证期最优
- experiment_id: auto_20260614_190800_exp_004
- model/horizon: ExtraTrees / 10D
- feature_top_k: 50
- target_label: avoid_loss_label
- strategy_family: ml_bull_full_participation_enhancement
- robust_score: 1.7048742668068624
- avg_excess_return_vs_buy_hold: 9.883543276349253e-05
- avg_missed_upside: 0.0005645124801649947
- avg_avoided_downside: 0.0009303600575551497

## 测试期最终表现
- test_total_return: 1.1127012123031261
- test_excess_return_vs_buy_hold: 0.003960528923004691
- test_max_drawdown: -0.17260282010543804
- test_sharpe: 2.332921486490729

## 是否跑赢买入持有
- 是

## 最优无杠杆策略（最终推荐，测试期）
- 策略名称: ml_bull_full_participation_enhancement
- 模型: ExtraTrees
- 特征数量(top_k): 50
- 标签/预测目标: avoid_loss_label
- 预测周期: 10D
- max_position_change(仓位平滑步长): 0.3
- 测试期 total_return: 1.112701
- buy_and_hold total_return: 1.108741
- excess_return: 0.003961
- max_drawdown: -0.172603
- buy_and_hold max_drawdown: -0.172603
- drawdown_improvement: 0.000000
- sharpe: 2.332921
- turnover(逐日平均): 0.000213
- average_position: 0.999788
- maximum_position: 1.000000
- maximum_position <= 1.0: 是
- 是否跑赢买入持有: 是
- 回撤是否低于买入持有: 否
