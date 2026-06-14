# Enhanced Index Strategy Summary

## Setup
- Initial capital: RMB 100,000
- Test period: 2025-01-01 to 2026-05-06
- Transaction cost rate: 0.001
- Signal timing: 收盘后观测预测/风险信号，应用于下一可交易收益。
- **No leverage: maximum position <= 1.0**

## Strategy Results
- Buy-and-hold total return: 1.1087
- Buy-and-hold max drawdown: -0.1726
- Best no-leverage ML strategy (ml_bull_full_participation_enhancement) total return: 1.1127
- ML strategy excess return: 0.0040
- ML strategy max drawdown: -0.1726
- Drawdown improvement vs buy-and-hold: 0.0000
- ML strategy sharpe: 2.3329
- ML strategy average position: 0.9998
- ML strategy maximum position: 1.0000
- ML strategy turnover (mean daily): 0.0002
- No-leverage ML strategy outperforms buy-and-hold: Yes

## Interpretation
- Position smoothing (max daily change limiter) 已应用，减少模型噪声导致的过度交易。
- 所有策略强制 maximum_position <= 1.0（无杠杆）。