# Auto Improve Report

## 搜索设置
- search_engine: optuna
- validation_folds: 2023H2, 2024H1, 2024H2
- test_period: 2025-01-01 to 2026-05-06
- max_runs: 8
- feature_top_k_candidates: [50, 80]
- families: ['ml_big_up_index_enhancement', 'ml_bull_full_participation_enhancement', 'ml_conservative_full_participation_enhancement', 'ml_ultra_conservative_full_participation_enhancement', 'ml_direct_signal_timing', 'ml_big_up_plus_115', 'ml_big_up_plus_120']
- target_labels: ['big_up_label', 'big_down_label', 'avoid_loss_label', 'reduce_position_worth_label']

## 可选模型依赖
- skipped_optional_models: []

## 验证期信号预筛选
- signal_leaderboard_path: /Users/luojingnan/Desktop/data_mining/Data-Mining/outputs/signal_leaderboard.csv
- candidate_count: 16
- top_candidate: ExtraTrees / 20D / top_k=50

## 验证期最优
- experiment_id: auto_20260602_190945_exp_001
- model/horizon: ExtraTrees / 10D
- feature_top_k: 50
- target_label: avoid_loss_label
- strategy_family: ml_bull_full_participation_enhancement
- robust_score: 1.7042203901154998
- avg_excess_return_vs_buy_hold: 0.00011047725537360886
- avg_missed_upside: 0.0006309257131255822
- avg_avoided_downside: 0.0010398141819734027

## 测试期最终表现
- test_total_return: 1.1131672500650112
- test_excess_return_vs_buy_hold: 0.004426566684889721
- test_max_drawdown: -0.17260282010543837
- test_sharpe: 2.3337838878560246

## 是否跑赢买入持有
- 是
