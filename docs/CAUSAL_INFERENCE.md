# Causal Inference

This document covers the three causal analyses embedded in the pipeline: the research question each answers, the identification strategy, the key assumptions, and how those assumptions are tested.

Read [`CAUSAL_DAG.md`](CAUSAL_DAG.md) first if you want the visual representation of the assumed causal structure before engaging with the formal methods.

---

## Why causal inference, not just prediction?

Most air quality pipelines stop at prediction: *will PM2.5 be dangerous tomorrow?* This project goes further by asking the causal question: *how many additional deaths and hospitalisations does a PM2.5 exceedance event actually cause?*

These are fundamentally different questions requiring different tools.

| Question | Tool | Consumer |
|---|---|---|
| Will PM2.5 exceed 25 µg/m³ at t+1? | Predictive model (LightGBM) | Hospitals, individuals, emergency managers |
| How many additional deaths does an episode cause? | Causal inference (DiD, IV) | Policymakers, regulators |
| Does the effect vary by region or age group? | Heterogeneous effects (Causal Forest) | Epidemiologists, targeted policy design |

A predictive alert says "prepare now." A causal estimate says "a policy reducing PM2.5 by X µg/m³ would prevent Y deaths per 100,000 residents per year."

---

## The identification problem

The core challenge is confounding. Observed correlation between PM2.5 levels and health outcomes is not sufficient to establish causation because the same factors that drive PM2.5 — industrial activity, economic structure, population density, urban heat islands — also directly affect health outcomes independently of PM2.5.

This backdoor path (U → PM2.5 and U → health, where U is unobserved) would bias any naive regression upward. Each of the three analyses uses a different strategy to block this path. See [`CAUSAL_DAG.md`](CAUSAL_DAG.md) for the full graphical representation.

---

## Analysis 1 — Difference-in-Differences (DiD)

**Research question:** What is the average increase in weekly respiratory mortality in a NUTS3 region in the days following a PM2.5 exceedance episode, compared to comparable regions that did not experience an episode in the same week?

**Implementation:** `src/causal/did_analysis.py` · `notebooks/03_causal_did.ipynb`

**Design:**

- **Treatment:** binary indicator — NUTS3 region × week where PM2.5 > 25 µg/m³ for ≥ 3 consecutive days
- **Control group:** NUTS3 regions in the same country and same week that did not exceed the threshold, selected to have similar pre-episode PM2.5 baseline (±20 % of treated region's 30-day average)
- **Outcome:** weekly respiratory deaths per 100,000 residents (EUROSTAT `hlth_cd_aro`)
- **Covariates:** ERA5 weekly averages (temperature, wind speed, precipitation, boundary layer height), NUTS3 fixed effects, week-of-year fixed effects

**Key assumption — parallel trends:** in the absence of treatment, the treated and control NUTS3 regions would have followed parallel mortality trends. This is the DiD identifying assumption and cannot be proven, only partially assessed.

**Validation:**
- Pre-trend test: event-study plot showing coefficients for t-4 to t-1 (pre-episode weeks) — these should be statistically indistinguishable from zero if parallel trends hold
- Falsification test: repeat analysis with a placebo outcome (injury-related deaths) that has no plausible PM2.5 pathway — the estimated effect should be zero
- Robustness: repeat with alternative threshold definitions (WHO 15 µg/m³, WHO annual 5 µg/m³)

**Reported outputs:**
- ATT (average treatment effect on the treated) with 95 % confidence interval, clustered standard errors at NUTS3 level
- Event-study plot: coefficients from t-4 to t+8 relative to the episode week
- Heterogeneity: ATT by country and season

---

## Analysis 2 — Instrumental Variables (IV)

**Research question:** What is the causal effect of PM2.5 concentration on weekly cardiovascular hospital admissions, accounting for unobserved confounders?

**Implementation:** `src/causal/iv_analysis.py` · `notebooks/04_causal_iv.ipynb`

**Instrument:** ERA5 boundary layer height (BLH), measured in metres above ground.

**Why BLH is a valid instrument:**

The instrument must satisfy two conditions:

1. **Relevance:** BLH must be strongly correlated with PM2.5 (first-stage). Temperature inversions — episodes of low BLH — trap pollutants near the ground, preventing vertical dispersion. This is a well-established atmospheric physics mechanism. First-stage F-statistic > 10 required; reported in all outputs.

2. **Exclusion restriction:** BLH must affect cardiovascular health *exclusively through its effect on PM2.5*, with no direct pathway. This is the maintained assumption. Its plausibility rests on the absence of any known biological mechanism by which atmospheric boundary layer height directly influences cardiovascular outcomes. Potential violations (e.g. if low BLH is correlated with cold temperature that independently increases cardiovascular risk) are addressed by including temperature as a control variable.

**Validation:**
- First-stage regression with F-statistic reported
- Sargan-Hansen overidentification test (when a secondary instrument is available — e.g. wind speed in a robustness check)
- Sensitivity analysis: repeat excluding summer months (when temperature-cardiovascular confounding is strongest)

**Reported outputs:**
- 2SLS estimate of the effect of a 10 µg/m³ increase in PM2.5 on weekly cardiovascular admissions per 100,000 residents, with 95 % confidence interval
- First-stage F-statistic
- Comparison: OLS estimate (confounded) vs. IV estimate (identified) to show direction and magnitude of confounding bias

---

## Analysis 3 — Heterogeneous Treatment Effects (Causal Forest)

**Research question:** Does the causal effect of PM2.5 on health outcomes vary across NUTS3 regions, seasons, or age groups? If so, by how much?

**Implementation:** `src/causal/heterogeneous_effects.py` · `notebooks/05_causal_heterogeneity.ipynb`

**Method:** Causal Forest (`econml.dml.CausalForestDML`) — a non-parametric method for estimating Conditional Average Treatment Effects (CATE) that allows the effect size to vary as a function of observable covariates.

**Design:**

- **Treatment:** continuous — weekly average PM2.5 concentration (µg/m³)
- **Outcome:** weekly all-cause mortality per 100,000 residents
- **Effect modifiers (X):** NUTS3 urbanisation index, median age, regional GDP per capita, season (winter/summer dummy), country fixed effect
- **Controls (W):** same ERA5 meteorological variables as DiD

**Why Causal Forest instead of a single average effect:**

A single DiD or IV estimate averages the effect across all regions, seasons, and demographic groups. If the effect is larger in winter (when baseline cardiovascular stress is higher), among elderly populations (whose respiratory reserve is lower), or in less wealthy regions (with worse healthcare access), a single average obscures the heterogeneity that is most actionable for policy targeting.

**Key assumption:** Conditional unconfoundedness — after conditioning on the observed controls W, the treatment (PM2.5) is as good as randomly assigned. This is weaker than the IV exclusion restriction but stronger than DiD parallel trends. The method does not handle unobserved confounders the way IV does.

**Validation:**
- Best linear predictor test (BLP): tests whether the estimated CATE is informative by checking if higher predicted CATE predicts higher actual outcomes
- RATE (Rank-Weighted Average Treatment Effect): measures the policy value of targeting the highest-CATE regions

**Reported outputs:**
- CATE choropleth map by NUTS3 region (dashboard causal panel)
- CATE by season: winter vs. summer point estimates with confidence intervals
- CATE by age group: < 65 vs. ≥ 65 years
- Feature importance: which effect modifiers explain the most CATE variation

---

## What causal estimates mean operationally

The `/health-impact` API endpoint returns the IV estimate of additional cardiovascular hospital admissions associated with a pollution episode of the magnitude predicted by the `/predict` endpoint.

**This is not a prediction of future health outcomes.** It is the historically estimated average effect of a PM2.5 episode of similar magnitude, averaged over the 2019–2023 period and the five countries in scope. It should be interpreted as: "episodes of this type have historically been associated with approximately X additional admissions per 100,000 residents in the affected NUTS3 region, after accounting for confounding."

For policy use, the appropriate framing is cost-benefit analysis of emission controls: "reducing average PM2.5 by Y µg/m³ across Region Z would, based on historical estimates, prevent approximately N additional deaths per year." This framing correctly treats the causal estimate as a structural parameter, not a forecast.

---

## Software and reproducibility

All analyses use fixed random seeds (`numpy.random.seed(42)`, `econml` `random_state=42`). Notebooks are designed to be run top-to-bottom with a single `jupyter nbconvert --to notebook --execute` call. Output cells are cleared before commit. All intermediate Parquet outputs are versioned via DVC.

Dependencies: `linearmodels==6.x`, `econml==0.15.x`, `dowhy==0.11.x`. See `requirements-causal.txt` for pinned versions.
