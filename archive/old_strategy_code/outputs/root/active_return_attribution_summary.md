# Active Return Attribution Summary

Objects analyzed:
- buy_hold
- ml_directional_alpha_extratrees_10d
- ml_cost_aware_overlay_conservative
- ml_index_enhanced_plus_120

## Interpretation
- ml_cost_aware_overlay_conservative is treated as a benchmark clone: benchmark_clone=Yes, avg_position=1.0000, days_below_full_exposure=0, tracking_error=0.0000.
- ml_directional_alpha_extratrees_10d active return comes from 55 reduced-exposure days, with missed-upside loss 2.28%, avoided-downside gain 1.57%, and incremental turnover cost 0.05%. The main drag is missed upside from reduced exposure on rising days; incremental trading cost is smaller.
- ml_index_enhanced_plus_120 positive excess return comes from enhanced exposure: max_position=1.2000, avg_position=1.0711, sum position-gap return=5.02%.

## ml_directional_alpha_extratrees_10d
- Selected row: ExtraTrees, horizon=10
- Strategy role: real ML timing strategy
- Benchmark clone: No
- Total return: 109.37%
- Excess return vs buy-and-hold: -1.28%
- Max drawdown: -16.70%
- Sum active return: -0.76%
- Sum position-gap return: -0.71%
- Total turnover cost: 0.15%
- Incremental turnover cost vs buy-and-hold: 0.05%
- Avg position: 0.9914
- Days below full exposure: 55
- Total turnover: 1.5000

Top 10 positive active-return dates:

| date | active_return | strategy_return | benchmark_return | position_gap | turnover_cost |
|---|---:|---:|---:|---:|---:|
| 2025-04-03 | 0.6544% | -12.4335% | -13.0879% | -0.0500 | 0.0000% |
| 2025-09-02 | 0.1300% | -2.4696% | -2.5995% | -0.0500 | 0.0000% |
| 2025-02-27 | 0.1284% | -2.4405% | -2.5690% | -0.0500 | 0.0000% |
| 2025-02-17 | 0.1215% | -2.3079% | -2.4294% | -0.0500 | 0.0000% |
| 2025-05-21 | 0.0733% | -1.3926% | -1.4659% | -0.0500 | 0.0000% |
| 2025-02-12 | 0.0637% | -1.2112% | -1.2750% | -0.0500 | 0.0000% |
| 2025-03-28 | 0.0576% | -1.1946% | -1.2522% | -0.0500 | 0.0050% |
| 2025-05-29 | 0.0556% | -1.0556% | -1.1111% | -0.0500 | 0.0000% |
| 2025-05-22 | 0.0476% | -0.9051% | -0.9527% | -0.0500 | 0.0000% |
| 2025-09-01 | 0.0458% | -0.8694% | -0.9151% | -0.0500 | 0.0000% |

Top 10 negative active-return dates:

| date | active_return | strategy_return | benchmark_return | position_gap | turnover_cost |
|---|---:|---:|---:|---:|---:|
| 2025-02-07 | -0.1342% | 2.4496% | 2.5838% | -0.0500 | 0.0050% |
| 2025-06-23 | -0.1328% | 2.5226% | 2.6554% | -0.0500 | 0.0000% |
| 2025-06-20 | -0.1317% | 2.4024% | 2.5341% | -0.0500 | 0.0050% |
| 2025-02-14 | -0.1288% | 2.4473% | 2.5761% | -0.0500 | 0.0000% |
| 2025-03-31 | -0.1048% | 1.9906% | 2.0954% | -0.0500 | 0.0000% |
| 2025-02-18 | -0.1028% | 1.9541% | 2.0570% | -0.0500 | 0.0000% |
| 2025-05-28 | -0.1003% | 1.9063% | 2.0066% | -0.0500 | 0.0000% |
| 2025-03-13 | -0.0994% | 1.8877% | 1.9870% | -0.0500 | 0.0000% |
| 2025-05-19 | -0.0867% | 1.6473% | 1.7340% | -0.0500 | 0.0000% |
| 2025-03-05 | -0.0840% | 1.5951% | 1.6790% | -0.0500 | 0.0000% |


## ml_cost_aware_overlay_conservative
- Selected row: ExtraTrees, horizon=10
- Strategy role: benchmark-clone strategy
- Benchmark clone: Yes
- Total return: 110.65%
- Excess return vs buy-and-hold: 0.00%
- Max drawdown: -17.26%
- Sum active return: 0.00%
- Sum position-gap return: 0.00%
- Total turnover cost: 0.10%
- Incremental turnover cost vs buy-and-hold: 0.00%
- Avg position: 1.0000
- Days below full exposure: 0
- Total turnover: 1.0000

Top 10 positive active-return dates:

| date | active_return | strategy_return | benchmark_return | position_gap | turnover_cost |
|---|---:|---:|---:|---:|---:|
| 2025-01-02 | 0.0000% | -4.8828% | -4.8828% | 0.0000 | 0.1000% |
| 2025-01-03 | 0.0000% | -0.6965% | -0.6965% | 0.0000 | 0.0000% |
| 2025-11-27 | 0.0000% | 1.7523% | 1.7523% | 0.0000 | 0.0000% |
| 2025-11-26 | 0.0000% | 1.2026% | 1.2026% | 0.0000 | 0.0000% |
| 2025-11-25 | 0.0000% | -0.9817% | -0.9817% | 0.0000 | 0.0000% |
| 2025-11-24 | 0.0000% | 1.7752% | 1.7752% | 0.0000 | 0.0000% |
| 2025-11-21 | 0.0000% | 2.1997% | 2.1997% | 0.0000 | 0.0000% |
| 2025-11-20 | 0.0000% | -4.8165% | -4.8165% | 0.0000 | 0.0000% |
| 2025-11-19 | 0.0000% | -0.6315% | -0.6315% | 0.0000 | 0.0000% |
| 2025-11-18 | 0.0000% | -2.0793% | -2.0793% | 0.0000 | 0.0000% |

Top 10 negative active-return dates:

| date | active_return | strategy_return | benchmark_return | position_gap | turnover_cost |
|---|---:|---:|---:|---:|---:|
| 2025-01-02 | 0.0000% | -4.8828% | -4.8828% | 0.0000 | 0.1000% |
| 2025-11-26 | 0.0000% | 1.2026% | 1.2026% | 0.0000 | 0.0000% |
| 2025-11-25 | 0.0000% | -0.9817% | -0.9817% | 0.0000 | 0.0000% |
| 2025-11-24 | 0.0000% | 1.7752% | 1.7752% | 0.0000 | 0.0000% |
| 2025-11-21 | 0.0000% | 2.1997% | 2.1997% | 0.0000 | 0.0000% |
| 2025-11-20 | 0.0000% | -4.8165% | -4.8165% | 0.0000 | 0.0000% |
| 2025-11-19 | 0.0000% | -0.6315% | -0.6315% | 0.0000 | 0.0000% |
| 2025-11-18 | 0.0000% | -2.0793% | -2.0793% | 0.0000 | 0.0000% |
| 2025-11-27 | 0.0000% | 1.7523% | 1.7523% | 0.0000 | 0.0000% |
| 2025-11-17 | 0.0000% | -0.7467% | -0.7467% | 0.0000 | 0.0000% |


## ml_index_enhanced_plus_120
- Selected row: GradientBoosting, horizon=5
- Strategy role: enhanced-exposure strategy
- Benchmark clone: No
- Total return: 120.05%
- Excess return vs buy-and-hold: 9.40%
- Max drawdown: -17.26%
- Sum active return: 4.82%
- Sum position-gap return: 5.02%
- Total turnover cost: 0.29%
- Incremental turnover cost vs buy-and-hold: 0.19%
- Avg position: 1.0711
- Days below full exposure: 25
- Total turnover: 2.9500

Top 10 positive active-return dates:

| date | active_return | strategy_return | benchmark_return | position_gap | turnover_cost |
|---|---:|---:|---:|---:|---:|
| 2026-03-05 | 0.6935% | 4.1613% | 3.4677% | 0.2000 | 0.0000% |
| 2026-03-09 | 0.5050% | 3.0302% | 2.5251% | 0.2000 | 0.0000% |
| 2026-02-13 | 0.4633% | 2.7797% | 2.3164% | 0.2000 | 0.0000% |
| 2025-11-21 | 0.4399% | 2.6396% | 2.1997% | 0.2000 | 0.0000% |
| 2025-10-20 | 0.4345% | 2.7069% | 2.2724% | 0.2000 | 0.0200% |
| 2025-07-18 | 0.3723% | 2.2338% | 1.8615% | 0.2000 | 0.0000% |
| 2026-01-16 | 0.3655% | 2.1927% | 1.8273% | 0.2000 | 0.0000% |
| 2025-10-30 | 0.3597% | 2.1582% | 1.7985% | 0.2000 | 0.0000% |
| 2025-11-24 | 0.3550% | 2.1303% | 1.7752% | 0.2000 | 0.0000% |
| 2025-11-27 | 0.3505% | 2.1028% | 1.7523% | 0.2000 | 0.0000% |

Top 10 negative active-return dates:

| date | active_return | strategy_return | benchmark_return | position_gap | turnover_cost |
|---|---:|---:|---:|---:|---:|
| 2025-11-20 | -0.9633% | -5.7798% | -4.8165% | 0.2000 | 0.0000% |
| 2025-08-26 | -0.7325% | -4.3949% | -3.6624% | 0.2000 | 0.0000% |
| 2026-02-27 | -0.6293% | -3.7758% | -3.1465% | 0.2000 | 0.0000% |
| 2026-03-18 | -0.5927% | -3.5559% | -2.9633% | 0.2000 | 0.0000% |
| 2026-03-02 | -0.5302% | -3.1809% | -2.6508% | 0.2000 | 0.0000% |
| 2025-08-13 | -0.5097% | -3.0583% | -2.5486% | 0.2000 | 0.0000% |
| 2026-03-16 | -0.4456% | -2.6738% | -2.2282% | 0.2000 | 0.0000% |
| 2025-11-18 | -0.4159% | -2.4951% | -2.0793% | 0.2000 | 0.0000% |
| 2025-12-03 | -0.3499% | -2.0996% | -1.7497% | 0.2000 | 0.0000% |
| 2025-07-14 | -0.2876% | -1.7255% | -1.4379% | 0.2000 | 0.0000% |

