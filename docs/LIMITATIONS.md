# Known Limitations

This document is mandatory reading before citing any output of this project in a policy document, academic paper, or public communication. The limitations here are not minor caveats — some of them bound the interpretability of results in fundamental ways.

---

## 1. Causal identification assumptions are maintained hypotheses, not verified facts

**DiD — parallel trends assumption.** The Difference-in-Differences analysis assumes that, in the absence of a PM2.5 episode, treated and control NUTS3 regions would have followed parallel mortality trends. This assumption is partially assessable through pre-trend tests (coefficients for t-4 to t-1 should be near zero) but cannot be proven. If treated regions were on a different trend trajectory before the episode — for example, because they are systematically more industrial or lower-income — the DiD estimate is biased.

**IV — exclusion restriction.** The Instrumental Variables analysis assumes that boundary layer height (BLH) affects cardiovascular health *exclusively* through PM2.5. The most plausible violation is the temperature channel: low BLH correlates with cold temperature episodes, and cold temperature independently increases cardiovascular stress. This is partially addressed by including temperature as a control variable, but residual confounding through uncontrolled temperature-related pathways cannot be ruled out.

**Causal Forest — conditional unconfoundedness.** The Causal Forest assumes that, conditional on the observed covariates (ERA5 meteorological variables, NUTS3 demographics), PM2.5 treatment is as good as randomly assigned. This is violated to the extent that unobserved confounders (industrial activity, socioeconomic status) simultaneously drive PM2.5 and health outcomes after conditioning. Unlike DiD and IV, Causal Forest provides no mechanism to address unobserved confounders.

**What this means in practice:** the three analyses should be interpreted as a triangulation rather than three independent confirmations. Consistent results across DiD, IV, and Causal Forest increase confidence. Divergent results signal a potential assumption violation that requires investigation.

---

## 2. Concept drift monitoring is approximate

Full concept drift detection — detecting a change in the conditional distribution P(Y|X) — requires refreshing ground truth labels on a near-weekly basis. The current monitoring implementation uses two proxies:
- **Label drift:** shift in the proportion of days exceeding 25 µg/m³
- **Feature drift:** PSI and KS tests on input feature distributions

These proxies detect marginal distribution shifts but will miss cases where the relationship between features and the outcome changes without a corresponding change in marginal distributions. For example, if a new industrial complex is built near several monitoring stations, the PM2.5-mortality relationship may change in that region without any shift in the overall feature distribution detectable by PSI.

---

## 3. Ground truth label lag

EEA monitoring data is available with a 24–48 hour lag. This means:
- Rolling recall in production is computed with up to 2 days' delay — not in real time
- Monitoring recall on health outcomes (mortality, hospital admissions) is infeasible in near-real-time: EUROSTAT weekly health data is available with a 4–8 week lag, WHO data with a 6–12 month lag

As a result, the monitoring system can quickly detect model degradation on PM2.5 exceedance prediction (the 24–48h label lag is acceptable), but cannot detect degradation of the causal health outcome estimates without a much longer delay.

---

## 4. Geographic scope is restricted to five Western European countries

The pipeline covers France, Spain, Belgium, the Netherlands, and Germany. It does not include:
- Central and Eastern Europe (Poland, Czech Republic, Romania, Hungary) — which have some of the highest PM2.5 exceedance rates in the EU
- Southern Europe (Italy, Greece) — different pollution profiles driven by Saharan dust and different industrial structures
- Northern Europe (Scandinavia) — different climate and atmospheric dynamics

**Implication for generalisation:** models trained on Western European stations may not generalise reliably to Eastern or Southern European contexts. The meteorological relationships (BLH, temperature, wind) and the health system response (hospitalisation rates, cause-of-death classification) may differ substantially. Do not apply the model or the causal estimates to regions outside the training scope without retraining on local data.

---

## 5. Health outcome data resolution mismatch

EUROSTAT and WHO health data are available at weekly NUTS3 granularity (aggregated over roughly 250,000–2,000,000 residents per region). EEA pollution data is daily at individual monitoring station level.

The spatial join from station-day to NUTS3-week introduces two sources of imprecision:
- **Spatial aggregation:** a NUTS3 region may contain multiple stations with heterogeneous PM2.5 levels; the aggregated weekly mean may not represent the exposure of the most affected sub-populations
- **Temporal aggregation:** a single high-pollution day within a week is averaged with low-pollution days; the acute spike that drives hospitalisation may be diluted in the weekly aggregate

**Implication:** the causal estimates likely understate the acute effect of peak pollution events, because the exposure measure is smoothed. Point estimates should be interpreted as effects of average weekly PM2.5, not of peak daily PM2.5.

---

## 6. Render.com free tier constraints

The production API is deployed on Render.com's free tier, which imposes:
- **Cold-start latency:** up to 30 seconds after a period of inactivity (> 15 minutes with no requests). The first request after cold start may time out for impatient clients.
- **Single instance:** no horizontal scaling. Under load (multiple simultaneous requests), queue time can increase significantly.
- **No persistent storage:** the model binary and threshold are loaded from the Docker image at startup, not from a live model registry. Updating the model requires a new Docker build and deploy.

This is appropriate for demonstration and research purposes. A production deployment with SLA requirements would require at minimum a paid tier with always-on instances and a live model registry integration (MLflow Model Registry with REST client in the API).

---

## 7. No real-time ingestion

The pipeline runs on a weekly batch schedule. It does not support:
- Streaming ingest from EEA monitoring stations
- Near-real-time prediction updates (predictions for the current day are based on data available as of the previous day's EEA upload)
- Intra-day refresh of the feature store

For a genuine operational early warning system — one that could trigger a health advisory within hours of a pollution event beginning — the ingestion and prediction layers would need to be redesigned around a streaming architecture (e.g. Kafka + Flink for ingestion, sub-hourly model serving).

---

## 8. Causal estimates are retrospective averages, not forward-looking forecasts

The `/health-impact` API endpoint returns the historically estimated average effect of a PM2.5 episode of a given magnitude, averaged over the 2019–2023 period and the five countries in scope.

**This is not a forecast.** It does not predict how many people will die or be hospitalised as a result of tomorrow's predicted pollution episode. It says: "episodes of this type, in these countries, over this historical period, were associated with approximately X additional deaths per 100,000 residents per week, after accounting for observed confounders."

For policy use, the correct framing is: "based on historical data, a pollution episode of this magnitude is associated with a health burden of approximately [N ± CI] additional deaths/admissions. Emergency preparedness should be scaled accordingly."

---

## 9. PM2.5 threshold sensitivity

The primary exceedance threshold (> 25 µg/m³, EU daily limit value) is a regulatory definition, not an epidemiological optimum. The WHO 2021 guidelines recommend 15 µg/m³ as the daily mean guideline — a threshold at which the model will produce far more exceedance events (more positive labels, different class balance, different recall/precision trade-offs).

A sensitivity analysis using the WHO 15 µg/m³ threshold is available in `notebooks/02_feature_validation.ipynb`. Results may differ substantially. The production model and all reported causal estimates are based on the EU 25 µg/m³ threshold unless otherwise specified.

---

## 10. Model performance degrades at longer lead times

The t+2 (48h) model consistently underperforms the t+1 (24h) model in recall by approximately 4–8 percentage points, depending on country and season. This is expected — atmospheric dynamics become less predictable at longer time horizons. Users of the `/predict` endpoint should weight t+1 predictions more heavily than t+2 predictions for operational decisions.
