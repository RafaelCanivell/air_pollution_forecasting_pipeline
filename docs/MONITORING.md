# Monitoring

This document covers the full monitoring design: offline evaluation metrics used during model development, and the production monitoring system for detecting model degradation and data drift.

---

## Design principles

1. **Monitor what you optimise.** The primary offline metric is recall (≥ 90 %). The primary production metric is rolling recall. They must be the same quantity.
2. **Distinguish operational health from model quality.** API latency and error rate tell you the service is running. Drift metrics and rolling recall tell you the model is still good. Both matter, but they are different signals requiring different responses.
3. **Design ground truth collection at training time, not after deployment.** If you cannot obtain labels for production data within an acceptable lag, you cannot monitor model quality. For this project, EEA station readings provide ground truth labels with a 24–48h lag — acceptable for a weekly batch pipeline.
4. **Alert on actions, not on numbers.** Every alert must have a defined response protocol: investigate, retrain, or escalate. An alert with no documented response is noise.

---

## Offline evaluation metrics

Used during model development to select the champion model and tune the decision threshold. Computed on the hold-out test set (temporal split — the last year of data, 2023).

| Metric | Target | Rationale |
|---|---|---|
| **Recall** | ≥ 90 % | Primary. A missed danger event (false negative) has higher public health cost than a false alarm. Optimise this first. |
| **Precision** | Reported, not optimised | Tracks false alarm rate. Too many false alarms cause alert fatigue in downstream systems. Monitor but do not sacrifice recall for precision. |
| **ROC-AUC** | Reported | Model discriminative power across all thresholds, independent of the tuned threshold. Useful for model comparison. |
| **F1 at tuned threshold** | Reported | Harmonic mean at the operational threshold. Only meaningful once the threshold is fixed. |
| **Brier score** | Reported | Calibration — are the predicted probabilities well-calibrated, or just discriminative? A model with recall = 92 % but severe probability miscalibration is unreliable for downstream uses of the probability score (e.g. the `/health-impact` endpoint uses the predicted probability to weight the causal estimate). |
| **Per-country recall** | ≥ 85 % for each country | Identifies regional weaknesses. A model with 92 % aggregate recall can have 70 % recall in one country hidden behind high performance in others. This must be caught before deployment. |

**Decision threshold tuning procedure:**

```python
thresholds = np.arange(0.30, 0.70, 0.01)
for t in thresholds:
    preds = (probabilities >= t).astype(int)
    rec = recall_score(y_val, preds)
    if rec >= 0.90:
        selected_threshold = t
        break  # lowest threshold achieving target recall

# Persist threshold as model artefact
mlflow.log_param("decision_threshold", selected_threshold)
```

The threshold is stored alongside the model binary in MLflow and loaded at inference time in `api/main.py`. It is never hardcoded.

---

## Production monitoring metrics

### Layer 1 — Operational health (service)

Tracked in real time. Failure here means the service is down or degraded, regardless of model quality.

| Metric | Alert threshold | Response | Tool |
|---|---|---|---|
| API latency p95 | > 500 ms | Investigate infrastructure | FastAPI middleware → `/metrics` endpoint |
| API error rate (5xx) | > 1 % over 5-min window | Check logs, restart if needed | GitHub Actions health check |
| Prediction volume | < 50 % of expected daily volume | Check Prefect flow, EEA data availability | Prefect flow alert |

### Layer 2 — Model quality (statistical)

Tracked per weekly batch, after the Prefect flow completes. These metrics require either ground truth labels or reference distribution comparison.

#### 2a. Model performance metrics (require ground truth)

Ground truth labels are derived from EEA station readings with a 24–48h lag. This makes recall monitoring feasible without external annotation and with a maximum 2-day delay.

| Metric | Alert threshold | Retraining trigger? |
|---|---|---|
| Rolling recall (7-day) | < 80 % | Yes — immediate retrain |
| Rolling precision (7-day) | Reported | No — informational only |

#### 2b. Data drift metrics (do not require ground truth)

Computed by comparing the current week's input feature distribution against the training baseline distribution, stored as a Parquet artefact at training time (`data/processed/monitoring/training_baseline.parquet`).

| Metric | Features monitored | Alert threshold | Retraining trigger? |
|---|---|---|---|
| **PSI (Population Stability Index)** | All input features | PSI > 0.1 → warning; PSI > 0.25 → alert | Yes, if PSI > 0.25 |
| **KS test** | `pm25_value`, `boundary_layer_height`, `temperature` | p-value < 0.01 | No — informational, triggers investigation |
| **KL divergence** | Predicted probability distribution vs. training period | KL > 0.1 | No — triggers investigation |
| **Label drift proxy** | Proportion of days exceeding 25 µg/m³ | ± 10 percentage points from training mean | No — triggers investigation |

**PSI interpretation:**

| PSI value | Interpretation | Action |
|---|---|---|
| < 0.1 | No significant drift | Continue |
| 0.1 – 0.25 | Moderate drift | Monitor closely, schedule review |
| > 0.25 | Significant drift | Trigger retraining |

**Why PSI over KL divergence as the primary drift metric:** PSI is more interpretable (a single number with established thresholds from the financial risk literature) and is computed per feature, making it easier to identify which specific features are drifting. KL divergence is used as a secondary check on the output distribution.

---

## Retraining trigger logic

Implemented in `src/pipeline/flow.py`, evaluated at the end of every weekly Prefect run.

```python
def should_retrain(
    new_data_available: bool,
    rolling_recall: float,
    max_psi: float,
    weeks_since_last_retrain: int,
    recall_threshold: float = 0.80,
    psi_threshold: float = 0.25,
    scheduled_retrain_weeks: int = 4,
) -> tuple[bool, str]:
    """
    Returns (retrain_flag, reason_string).
    Priority: quality degradation > drift > scheduled.
    """
    if not new_data_available:
        return False, "no new data"

    if rolling_recall < recall_threshold:
        return True, f"recall degradation: {rolling_recall:.2f} < {recall_threshold}"

    if max_psi > psi_threshold:
        return True, f"feature drift: max PSI {max_psi:.3f} > {psi_threshold}"

    if weeks_since_last_retrain >= scheduled_retrain_weeks:
        return True, f"scheduled retrain: {weeks_since_last_retrain} weeks since last"

    return False, "no trigger met"
```

The reason string is logged to MLflow as a tag on the training run, creating an audit trail of why each model was retrained.

---

## Champion selection

After retraining, the new model is not deployed automatically. It must pass the champion selection gate:

```python
def select_champion(
    new_model_recall: float,
    current_model_recall: float,
    new_model_psi: float,
    recall_min: float = 0.90,
    psi_max: float = 0.25,
) -> bool:
    """
    Deploy new model only if it meets absolute threshold
    AND does not introduce new drift relative to training baseline.
    """
    meets_recall = new_model_recall >= recall_min
    within_drift = new_model_psi <= psi_max
    improves_or_holds = new_model_recall >= current_model_recall - 0.02  # 2pp tolerance

    return meets_recall and within_drift and improves_or_holds
```

If the new model fails champion selection, the current production model is retained and an alert is raised for manual investigation.

---

## Monitoring dashboard

All metrics are exposed via `GET /metrics` (JSON) and logged to MLflow as a dedicated monitoring run separate from training runs. The Plotly Dash dashboard (`dashboard/app.py`) includes a model performance panel showing:

- Rolling recall (7-day and 30-day) over time
- PSI per feature, colour-coded by alert level
- Prediction volume vs. expected volume
- Decision threshold history (has it been adjusted by retraining?)

---

## Known monitoring limitations

See [`LIMITATIONS.md`](LIMITATIONS.md) for the full list. Monitoring-specific limitations:

- **Concept drift monitoring is approximate.** Full P(Y|X) shift detection requires weekly ground truth labels. The label drift proxy (exceedance rate shift) is a necessary but not sufficient condition for concept drift. A situation where the feature-outcome relationship changes without a shift in marginal distributions would not be detected.
- **Ground truth lag for health outcomes.** Mortality and hospitalisation data (EUROSTAT, WHO) are available with a lag of weeks to months. Monitoring recall on health outcomes in near-real-time is not feasible. The 24–48h EEA label lag applies only to the PM2.5 exceedance prediction, not to the causal health outcome estimates.
- **Render.com free tier.** Cold-start latency makes the p95 latency alert unreliable for the first request after a period of inactivity. The alert is suppressed for the first request after a 30-minute gap.
