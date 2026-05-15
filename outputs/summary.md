# Enhanced Index Strategy Summary

## Setup
- Initial capital: RMB 100,000
- Test period: 2025-01-01 to 2026-05-06
- Transaction cost rate: 0.0010
- Signal timing: predictions and regime filters are observed after close on date t and applied to the next tradable return.

## Prediction Interpretation
- Best test ordinary direction AUC: 0.5562
- Best test clean-direction AUC: 0.5742
- Best test big-up AUC: 0.6163
- Ordinary direction AUC and clean-direction AUC are below 0.60, so the model is not used as an aggressive 0/1 market-timing engine.
- Big-up AUC is above 0.60, so the decision layer focuses on strong-upside participation and risk-aware exposure adjustment.

## Strategy Results
- Buy-and-hold total return: 1.1087
- Buy-and-hold max drawdown: -0.1726
- Best no-leverage strategy: ml_ultra_conservative_full_participation_enhancement
- Whether best no-leverage strategy is benchmark clone: False
- Best real no-leverage ML strategy: ml_ultra_conservative_full_participation_enhancement
- ML strategy total return: 1.0861
- ML strategy excess return: -0.0226
- ML strategy max drawdown: -0.1726
- Best enhanced-exposure strategy: ml_big_up_plus_120
- Best enhanced-exposure excess return: -0.0915
- Any real no-leverage ML strategy outperforms buy-and-hold: No

## Interpretation
No-leverage and enhanced-exposure strategies are reported separately. If no-leverage ML enhancement fails to outperform buy-and-hold, the main reason is that the test period was strongly upward and even mild defensive exposure cuts can miss upside. If an enhanced-exposure strategy outperforms, that result should be interpreted separately because it allows exposure above 1.00.
