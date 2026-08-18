# V3 Common Evaluation Result — v3-evaluator-001

The common evaluator regenerated EXP-001 through EXP-004 under one chronological scoring contract and verified identical realized decision-date samples inside every comparable fold/target lane.

## Main conclusion

The initial feature/model set has **not yet demonstrated robust predictive edge**. Random forest is the strongest of the tested models in aggregate, but being best among the initial candidates is not sufficient for champion status.

### Classification aggregate

| Experiment | Model | Mean Brier | Mean log loss | Mean ECE | Mean relative Brier improvement |
| --- | --- | ---: | ---: | ---: | ---: |
| EXP-001 | Logistic | 0.2808 | 0.8626 | 0.2327 | -62.1% |
| EXP-003 | Histogram gradient boosting | 0.3207 | 0.9723 | 0.3026 | -82.8% |
| EXP-004 | Random forest | **0.2320** | **0.6622** | **0.1840** | **-29.3%** |

Even the best classification model is worse on average than the simple fold base-rate Brier benchmark.

### Return-regression aggregate

| Experiment | Model | Mean MAE | Mean RMSE | Mean Spearman rank correlation |
| --- | --- | ---: | ---: | ---: |
| EXP-002 | Ridge | 4.13% | 5.28% | -0.1540 |
| EXP-003 | Histogram gradient boosting | 3.83% | 4.78% | -0.0707 |
| EXP-004 | Random forest | **3.25%** | **4.12%** | -0.1001 |

All three return models have negative mean rank correlation across the fold/horizon cells, another reason not to treat current predictions as a validated trading edge.

### 20-session drawdown regression

Random forest also has lower aggregate drawdown error than gradient boosting (MAE about 2.18% versus 2.84%), but its mean rank correlation is approximately zero.

## Consequence for the roadmap

- Do **not** select a champion yet.
- Do **not** begin tactical sizing based on these results.
- V3-010 should formalize the scoreboard and mark the initial set as not promotion-ready.
- The next research value should come from independent information families (starting with VIX in V3-011/V3-012), evaluated through controlled ablations rather than retuning the existing models.

Exact common predictions and metric outputs are preserved in GitHub Actions run `32198316399`; their SHA-256 hashes are recorded in `evaluator_manifest.json`.
