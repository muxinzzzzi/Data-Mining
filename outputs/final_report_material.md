# Final Report Material

## Research Question
使用微盘股指数日线 OHLCV 与 5 分钟分钟线辅助特征，构建无杠杆机器学习指数增强策略。测试区间固定为 2025-01-01 到 2026-05-06，初始资金 100,000 元。

## Leakage Control (防数据泄漏)
- 固定验证折（2023H2/2024H1/2024H2）与固定测试期，无随机划分。
- walk-forward 训练严格要求 `target_end_date < chunk_start`，标签/特征均不使用未来信息。
- 特征选择、预处理、模型与参数搜索只用训练/验证数据；测试期只用于最终评估。
- rolling 风险特征与未来收益标签构造时已做 shift，避免当日泄漏。

## Improvements Made (本轮改进)
- 新增真正的仓位平滑（max daily change limiter），限制单日仓位变化幅度，减少模型短期噪声造成的过度交易与回撤；该步长 `max_position_change` 纳入 Optuna 搜索。
- 搜索目标 robust_score 同时考虑超额收益、回撤、换手、机会成本与跨折稳定性。
- 安装并启用 Optuna / LightGBM / XGBoost，max_runs 提升到 >=50。
- 所有候选策略强制 maximum_position <= 1.0（无杠杆），已移除 1.15/1.20 杠杆族出最终推荐。

## Best No-Leverage Strategy Results (测试期)
| Metric | Value |
| --- | --- |
| Strategy | ml_bull_full_participation_enhancement |
| Model | ExtraTrees |
| Horizon | 10D |
| Decision Target Label | avoid_loss_label |
| Feature top_k | 50 |
| max_position_change | 0.3 |
| Strategy Total Return | 1.1127 |
| Buy & Hold Total Return | 1.1087 |
| Excess Return | 0.0040 |
| Strategy Max Drawdown | -0.1726 |
| Buy & Hold Max Drawdown | -0.1726 |
| Drawdown Improvement | 0.0000 |
| Sharpe | 2.3329 |
| Turnover (mean daily) | 0.0002 |
| Average Position | 0.9998 |
| Maximum Position | 1.0000 |

## Constraint Verification
- Maximum Position <= 1.0: **Yes** (max_position=1.0000)
- Test set not used for tuning: **Yes**（仅用于最终评估）。
- Outperforms buy-and-hold: **Yes**。
- Max drawdown lower than buy-and-hold: **No**。