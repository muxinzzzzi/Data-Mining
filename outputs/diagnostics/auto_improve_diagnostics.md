# Auto Improve Diagnostics

## 概览
- total_runs: 50
- success_runs: 50
- failed_runs: 0

## 防泄漏说明
- 多折验证仅使用 2023H2、2024H1、2024H2。
- 测试期固定为 2025-01-01 到 2026-05-06。
- robust_score、模型选择、参数搜索、特征选择均不使用测试期信息。
- 每个 walk-forward chunk 都只使用 target_end_date < chunk_start 的训练样本。

## 组合有效性统计
| model_name | decision_target_label | family | run_count | success_count | avg_robust_score |
| --- | --- | --- | --- | --- | --- |
| ExtraTrees | avoid_loss_label | ml_bull_full_participation_enhancement | 2 | 2 | 1.607654077412936 |
| RandomForest | big_up_label | ml_conservative_full_participation_enhancement | 2 | 2 | 1.5984738576241675 |
| ExtraTrees | big_up_label | ml_bull_full_participation_enhancement | 1 | 1 | 1.5104338880190096 |
| ExtraTrees | reduce_position_worth_label | ml_bull_full_participation_enhancement | 1 | 1 | 1.5104338880190096 |
| HistGradientBoosting | avoid_loss_label | ml_ultra_conservative_full_participation_enhancement | 1 | 1 | 1.5104338880190096 |
| HistGradientBoosting | big_down_label | ml_ultra_conservative_full_participation_enhancement | 1 | 1 | 1.5104338880190096 |
| LightGBM | avoid_loss_label | ml_bull_full_participation_enhancement | 1 | 1 | 1.5104338880190096 |
| RandomForest | big_down_label | ml_ultra_conservative_full_participation_enhancement | 1 | 1 | 1.5104338880190096 |
| RandomForest | reduce_position_worth_label | ml_bull_full_participation_enhancement | 1 | 1 | 1.5104338880190096 |
| XGBoost | big_up_label | ml_ultra_conservative_full_participation_enhancement | 1 | 1 | 1.5104338880190096 |
| RandomForest | big_up_label | ml_bull_full_participation_enhancement | 3 | 3 | 1.5091364226491548 |
| RandomForest | avoid_loss_label | ml_ultra_conservative_full_participation_enhancement | 1 | 1 | 1.5076364867148453 |

## 信号预筛选前列候选
| feature_top_k | model_name | horizon | avg_selection_score | avg_clean_direction_auc | avg_big_up_auc | avg_big_down_auc | avg_trade_auc | avg_return_correlation | selection_score_std | fold_count | signal_robustness | candidate_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | ExtraTrees | 20 | 0.6546732343347661 | 0.7505834674891622 | 0.7207000687523718 | 0.6743474996034093 | 0.6504854165973247 | 0.2201326918229369 | 0.1657626677819746 | 3 | 0.6933414311071042 | 0 |
| 50 | RandomForest | 20 | 0.6508765595553093 | 0.7467924835666994 | 0.6860476601031639 | 0.7181186168298347 | 0.6783262881971633 | 0.09841332544044856 | 0.15896720599616326 | 3 | 0.6908214275567351 | 1 |
| 50 | LightGBM | 20 | 0.6512195063961154 | 0.7220868663837844 | 0.6623172012892323 | 0.7757127239015694 | 0.6633613503771323 | 0.1467030106669223 | 0.15636176351313005 | 3 | 0.6892799462922864 | 2 |
| 50 | XGBoost | 20 | 0.6367535801131278 | 0.7172160532964541 | 0.6959828089363601 | 0.6971163538847689 | 0.6329622096910476 | 0.1146664289488201 | 0.1890437223086835 | 3 | 0.6642446346014602 | 3 |
| 50 | HistGradientBoosting | 20 | 0.6181732236436206 | 0.7050127882493369 | 0.6103741188259331 | 0.7220773155951502 | 0.6097241331387674 | 0.10370460536007224 | 0.1637484392201369 | 3 | 0.6462746474109734 | 4 |
| 50 | ExtraTrees | 10 | 0.5987826145776723 | 0.6808195057536585 | 0.7285151977685479 | 0.5722554194776417 | 0.5916940692301059 | 0.10488330037274059 | 0.18287439925755444 | 3 | 0.6285594348128185 | 5 |
| 50 | ExtraTrees | 5 | 0.5709322075275767 | 0.6617675381578314 | 0.6766281110797415 | 0.5272173148036288 | 0.5928635819654727 | 0.09680922232546302 | 0.13862245424314668 | 3 | 0.612116412346783 | 6 |
| 50 | RandomForest | 10 | 0.583700349318128 | 0.6418500030879312 | 0.6670540399485075 | 0.5841312121268548 | 0.5847965209807854 | 0.1598881885938244 | 0.1962829634066442 | 3 | 0.6034534265805301 | 7 |
| 80 | LightGBM | 20 | 0.6669848027115867 | 0.7464471201916166 | 0.7050150984144808 | 0.7356119817931237 | 0.6695243584196239 | 0.2607205851471916 | 0.152366998535467 | 3 | 0.7102290226025092 | 8 |
| 80 | RandomForest | 20 | 0.6727718775355508 | 0.7751667829537379 | 0.7221916742511506 | 0.7309857604932003 | 0.6600075327191539 | 0.1648054045123066 | 0.1748352651377671 | 3 | 0.7089553206405735 | 9 |
| 80 | XGBoost | 20 | 0.6613407733282226 | 0.756904449960635 | 0.7160260841263156 | 0.7387453475069559 | 0.6569360869217397 | 0.20301627127589575 | 0.15012634353074278 | 3 | 0.7054948050378722 | 10 |
| 80 | HistGradientBoosting | 20 | 0.6476756236349427 | 0.7359225818502382 | 0.6645615836420969 | 0.7290207115067288 | 0.6398465135480917 | 0.2605244593228402 | 0.15915868553320425 | 3 | 0.6835258012025744 | 11 |
