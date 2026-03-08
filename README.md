#  (Work in progress) Air Quality Prediction Pipeline — NOAA + EEA
### *Predicting dangerous pollution events using large-scale atmospheric and environmental data across Europe*

---

>  **English version below** |  **Versión en español más abajo**

---

##  English

### Business Case & Problem Statement

Air pollution is the single largest environmental health risk in Europe, responsible for over **400,000 premature deaths per year** across the EU according to the European Environment Agency. Despite significant progress in reducing emissions over recent decades, pollutant concentrations still exceed both EU legal limits and the stricter WHO guidelines in many urban and industrial areas — particularly in Central and Eastern Europe, the Po Valley in Italy, and heavily trafficked Western European cities.

The economic cost is equally severe: lost working days, hospital admissions, and long-term chronic disease place hundreds of billions of euros of pressure on European healthcare systems annually, with the burden falling disproportionately on lower-income populations living near industrial zones, highways, and ports.

Despite the availability of dense monitoring networks, most public-facing tools offer only *current* air quality readings — they do not **predict** when conditions will become dangerous. A 24 to 48-hour advance warning of a high-pollution event allows individuals, hospitals, and municipal governments to take preventive action: issuing health advisories, adjusting industrial operations, activating low-emission zones, or simply advising vulnerable populations to stay indoors.

This project addresses that gap by building an **end-to-end machine learning pipeline** that ingests historical atmospheric and pollution data across Europe, engineers predictive features at scale, and serves real-time forecasts through a production-grade API.

---

### Why Combine NOAA GHCN and EEA Air Quality Data?

These two datasets are individually powerful but tell incomplete stories on their own.

**EEA Air Quality e-Reporting** provides ground-level measurements of five key pollutants — PM2.5, PM10, ozone (O₃), nitrogen dioxide (NO₂), and sulphur dioxide (SO₂) — collected from thousands of official monitoring stations across all EU member states and several neighboring countries. It tells us *what the air contains*, but not *why* conditions changed or *what will happen next*. Data is available hourly, downloadable by country and pollutant via API, and delivered natively in Parquet format.

**NOAA Global Historical Climatology Network (GHCN)** provides decades of surface weather observations — temperature, precipitation, wind speed, humidity, and atmospheric pressure — from stations worldwide, including extensive coverage across Europe through its integration of the European Climate Assessment dataset (ECA&D). It tells us about the *atmospheric conditions* that drive pollutant behavior: temperature inversions that trap particles near the ground, wind patterns that disperse or concentrate pollutants across national borders, and precipitation events that temporarily clean the air.

Combining both datasets allows the model to learn the **physical relationship between weather and pollution**, which is exactly what drives real-world air quality dynamics. Neither dataset alone is sufficient for accurate forecasting; together, they enable a model that understands both the *source signal* (pollutant levels) and the *driving conditions* (meteorology).

The transboundary nature of European air pollution — where industrial emissions from one country affect air quality in neighboring ones — makes the multi-country, multi-source approach especially relevant. This is also what justifies Apache Spark: the combined historical dataset spans decades, dozens of countries, and hundreds of millions of records that do not fit comfortably in memory.

---

### Why PM2.5 as the Prediction Target?

The pipeline downloads and processes all five EEA pollutants — PM2.5, PM10, NO₂, O₃, and SO₂. In the modelling phase, **PM2.5 is chosen as the primary prediction target**, but the other four pollutants are retained as input features rather than discarded. This is a deliberate design choice grounded in both epidemiology and machine learning.

**PM2.5 as target — the public health case:**

PM2.5 (fine particulate matter with diameter ≤ 2.5 µm) is the pollutant with the strongest and most consistent evidence of harm across medical literature. Unlike PM10, which is largely filtered in the upper respiratory tract, PM2.5 particles penetrate deep into the lungs and cross into the bloodstream, causing cardiovascular disease, respiratory illness, and neurological damage. The WHO 2021 Air Quality Guidelines revised the PM2.5 annual mean guideline down to 5 µg/m³ — the most stringent update in two decades — precisely because the evidence base for harm at low concentrations is now overwhelming.

For a predictive warning system, PM2.5 exceedance events are the most actionable signal: they are the threshold that triggers health advisories, activates emergency protocols in hospitals, and determines legal compliance for industrial operators. Predicting PM10 or SO₂ alone would be less operationally meaningful.

**PM2.5 as target — the modelling case:**

Beyond the public health rationale, PM2.5 has the best statistical properties for learning: strong temporal autocorrelation (yesterday's PM2.5 is highly predictive of tomorrow's), clear seasonal patterns driven by heating seasons and temperature inversions, and a well-documented physical relationship with meteorological variables. This makes lag features and rolling windows particularly effective, and SHAP values particularly interpretable.

**The other pollutants as features — not wasted:**

PM10, NO₂, O₃, and SO₂ are not discarded — they are used as input features alongside weather variables, with the following physical justifications:

| Feature | Role | Physical rationale |
|---|---|---|
| `pm10_value` | Same-day feature + t-1 lag | PM10 is a superset of PM2.5 — strong collinear predictor. High PM10 days almost always imply elevated PM2.5. |
| `no2_value` | Same-day feature + t-1 lag | NO₂ is a proxy for traffic and combustion intensity — the same processes that emit PM2.5. High NO₂ signals high emission activity. |
| `o3_value` | Same-day feature + t-1 lag | O₃ tends to **anticorrelate** with PM2.5 due to photochemical competition. Low O₃ in summer can signal stagnant air conditions that accumulate particles. The inverted signal is still informative. |
| `so2_value` | Same-day feature + t-1 lag | SO₂ indicates industrial activity that co-emits fine particles. Persistent SO₂ episodes often precede PM2.5 exceedance events in industrial regions. |

Same-day pollutant values are not leaky: at prediction time (end of day *t*), all five pollutant readings for day *t* are already known from the monitoring network. The model uses them to predict whether PM2.5 will exceed the threshold tomorrow (*t+1*) or the day after (*t+2*).

This design — one focused target, multiple co-pollutants as features — is more defensible than training five separate models or ignoring four of the five downloaded pollutants. It reflects the real structure of the atmosphere: pollutants do not behave independently, and the interactions between them carry predictive signal.

---

### Expected Outcomes & Outputs

| Output | Description |
|---|---|
| **Processed feature store** | Parquet dataset partitioned by country, year, and pollutant — clean, spatially joined, and ready for modeling |
| **Aggregation tables** | City-level summaries: days above EU/WHO thresholds per year, seasonal patterns, cross-border pollution correlations |
| **Trained ML models** | LightGBM and XGBoost classifiers predicting PM2.5 threshold exceedance at 24h (t+1) and 48h (t+2) horizons |
| **SHAP analysis** | Interpretability report identifying which features most strongly drive pollution events by region and season |
| **Production API** | FastAPI endpoint serving predictions with alert level, probability, decision threshold, and top contributing factors |
| **Orchestrated pipeline** | Prefect DAG automating the full flow from ingestion to model retraining, scheduled weekly |
| **Interactive dashboard** | Plotly Dash application showing Europe-wide PM2.5 trends, seasonal heatmaps, and model performance metrics |

---

### Modelling Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Prediction target | PM2.5 exceedance (binary) | Highest health impact; strong autocorrelation; actionable threshold |
| Prediction horizons | t+1 (24h) and t+2 (48h) | Operationally meaningful for health advisories and industrial planning |
| Co-pollutants | Features, not targets | Retain NO₂, O₃, PM10, SO₂ as inputs — they encode emission and atmospheric state |
| Priority metric | Recall (≥ 90%) | Missing a danger event (false negative) is worse than a false alarm |
| Class imbalance | `scale_pos_weight` in LightGBM/XGBoost | Adjusts for the fact that danger days are a minority of observations |
| Validation strategy | Strict temporal split | Random splits cause data leakage via lag features |
| Decision threshold | Tuned per model | Threshold set to achieve ≥ 90% recall on validation set, not fixed at 0.5 |

---

### Methodology

**Phase 1 — Ingestion & Exploratory Analysis**
Raw data is downloaded programmatically from the EEA Air Quality Download Service and NOAA GHCN. EDA documents data coverage, missing value patterns, distribution shapes, class imbalance, seasonal structure, cross-pollutant correlations, and autocorrelation of PM2.5 across countries.

**Phase 2 — Large-Scale Processing with Apache Spark**
Three PySpark jobs handle cleaning, spatial joining (Haversine nearest-neighbor between EEA and NOAA stations, max 50 km), and feature engineering: PM2.5 and co-pollutant lags, rolling windows, calendar features, and derived meteorological indices. Outputs are partitioned Parquet files.

**Phase 3 — Pipeline Orchestration with Prefect**
All steps are wrapped in a Prefect flow with dependency management, schema validation gates, retry logic with exponential backoff, and weekly scheduling. The pipeline runs incrementally — only new data is downloaded on each run. A flag file triggers model retraining only when new data arrived.

**Phase 4 — Predictive Modeling**
Binary classification (PM2.5 > 25 µg/m³ at t+1 and t+2) using LightGBM (primary), XGBoost (comparative), and Logistic Regression (interpretable baseline). Decision threshold tuned to achieve ≥ 90% recall. SHAP values provide full feature interpretability. Country-level performance breakdown identifies regional weaknesses.

**Phase 5 — Production Deploy**
FastAPI application containerised with Docker (multi-stage build, non-root user). CI/CD via GitHub Actions: tests on every PR, automatic deploy to Render.com on push to main. Endpoints: `/predict`, `/predict/batch`, `/health`, `/metrics`.

---

### Tech Stack

| Layer | Tool |
|---|---|
| Ingestion | Python, `airbase`, `requests` |
| Processing | Apache Spark (PySpark) |
| Orchestration | Prefect |
| Modeling | LightGBM, XGBoost, scikit-learn, SHAP |
| Visualization | Plotly Dash |
| API | FastAPI |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Deploy | Render.com (free tier) |

---

### Project Structure

```
noaa-eea-air-quality-europe/
├── data/                        # excluded from git (.gitignore)
│   ├── raw/
│   │   ├── eea/                 # verified + unverified Parquet
│   │   └── noaa/                # yearly CSV.gz + station metadata
│   └── processed/
│       ├── eea/                 # cleaned EEA, partitioned
│       ├── noaa/                # cleaned NOAA wide format, partitioned
│       ├── features/            # feature store (final ML input)
│       └── aggregations/        # city-level yearly summaries
├── notebooks/
│   ├── 01_eda.ipynb             # EDA: distributions, seasonality, correlations
│   └── 02_feature_validation.ipynb  # feature store QA + quick model experiment
├── src/
│   ├── ingestion/
│   │   ├── download_eea.py
│   │   ├── download_noaa.py
│   │   └── validate_downloads.py
│   ├── spark/
│   │   ├── spark_clean_eea.py
│   │   ├── spark_clean_noaa.py
│   │   └── spark_join_features.py
│   ├── model/
│   │   ├── train.py
│   │   └── evaluate.py
│   └── pipeline/
│       └── flow.py              # Prefect DAG
├── api/
│   └── main.py                  # FastAPI application
├── dashboard/
│   └── app.py                   # Plotly Dash dashboard
├── tests/
│   └── test_api.py
├── models/                      # saved model artifacts (excluded from git)
├── Dockerfile                   # API production image
├── Dockerfile.pipeline          # Pipeline image (Spark + Prefect)
├── docker-compose.yml
├── prefect.yaml
├── requirements-ingestion.txt
├── requirements-spark.txt
├── requirements-model.txt
├── requirements-api.txt
├── requirements-dashboard.txt
└── requirements-pipeline.txt
```

---

### Quick Start

```bash
# 1. Install ingestion dependencies
pip install -r requirements-ingestion.txt

# 2. Download data (dry-run first to check sizes)
python src/ingestion/download_eea.py --dry-run
python src/ingestion/download_noaa.py --dry-run

# 3. Download for real
python src/ingestion/download_eea.py
python src/ingestion/download_noaa.py

# 4. Validate downloads
python src/ingestion/validate_downloads.py

# 5. Run Spark jobs
pip install -r requirements-spark.txt
spark-submit src/spark/spark_clean_eea.py
spark-submit src/spark/spark_clean_noaa.py
spark-submit src/spark/spark_join_features.py

# 6. Train models
pip install -r requirements-model.txt
python src/model/train.py
python src/model/evaluate.py

# 7. Run API locally
pip install -r requirements-api.txt
uvicorn api.main:app --reload
# Swagger UI → http://localhost:8000/docs

# 8. Run dashboard
pip install -r requirements-dashboard.txt
python dashboard/app.py
# Dashboard → http://localhost:8050
```

---

### Data Sources

- **EEA Air Quality e-Reporting:** https://www.eea.europa.eu/en/datahub/datahubitem-view/778ef9f5-6293-4846-badd-56a29c70880d
- **NOAA GHCN Daily (AWS public bucket):** https://registry.opendata.aws/noaa-ghcn

---
---

##  Español

### (Trabajo en curso) Caso de Negocio y Planteamiento del Problema

La contaminación del aire es el mayor riesgo ambiental para la salud en Europa, responsable de más de **400.000 muertes prematuras al año** en la UE según la Agencia Europea de Medio Ambiente. A pesar del progreso significativo en la reducción de emisiones, las concentraciones de contaminantes siguen superando tanto los límites legales de la UE como las directrices más estrictas de la OMS en muchas zonas urbanas e industriales — especialmente en Europa Central y del Este, el Valle del Po en Italia y las ciudades del oeste con alto tráfico.

La mayoría de las herramientas públicas solo ofrecen lecturas *actuales* de calidad del aire — no **predicen** cuándo las condiciones se volverán peligrosas. Un aviso de 24 a 48 horas antes de un evento de alta contaminación permite que personas, hospitales y gobiernos municipales tomen medidas preventivas: emitir alertas sanitarias, ajustar operaciones industriales, activar zonas de bajas emisiones, o simplemente indicar a poblaciones vulnerables que permanezcan en interiores.

---

### ¿Por Qué Combinar los Datos de NOAA GHCN y EEA?

**EEA Air Quality e-Reporting** nos dice *qué contiene el aire* — niveles de PM2.5, PM10, O₃, NO₂ y SO₂ — pero no *por qué cambiaron las condiciones* ni *qué pasará después*.

**NOAA GHCN** nos habla de las *condiciones atmosféricas* que determinan el comportamiento de los contaminantes: inversiones térmicas que atrapan partículas cerca del suelo, patrones de viento que dispersan o concentran contaminantes a través de fronteras, y eventos de lluvia que limpian temporalmente el aire.

Combinar ambos datasets permite que el modelo aprenda la **relación física entre el clima y la contaminación**. El carácter transfronterizo de la contaminación en Europa justifica además el procesamiento distribuido con Apache Spark: el dataset combinado abarca décadas, docenas de países y cientos de millones de registros.

---

### ¿Por Qué PM2.5 como Variable Objetivo?

El pipeline descarga y procesa los cinco contaminantes de EEA. En la fase de modelado, **PM2.5 es el target principal**, pero los otros cuatro contaminantes se retienen como *features de entrada*, no se descartan.

**PM2.5 como target — el caso de salud pública:**

PM2.5 (partículas finas de diámetro ≤ 2,5 µm) es el contaminante con mayor evidencia científica de daño. A diferencia del PM10, que queda filtrado en el tracto respiratorio superior, las partículas finas penetran en los pulmones y cruzan al torrente sanguíneo, causando enfermedades cardiovasculares, respiratorias y neurológicas. La OMS revisó en 2021 su directriz anual de PM2.5 a 5 µg/m³ — la actualización más estricta en dos décadas — precisamente porque la evidencia de daño a bajas concentraciones es ahora abrumadora.

Para un sistema de alertas predictivo, el umbral de PM2.5 es la señal más accionable: es el que activa avisos sanitarios, protocolos de emergencia hospitalaria y obligaciones de cumplimiento para operadores industriales.

**PM2.5 como target — el caso del modelado:**

PM2.5 tiene las mejores propiedades estadísticas para aprender: fuerte autocorrelación temporal, estacionalidad clara y relación bien documentada con variables meteorológicas. Los lag features y ventanas móviles son especialmente efectivos, y los valores SHAP son especialmente interpretables.

**Los otros contaminantes como features — no desperdiciados:**

PM10, NO₂, O₃ y SO₂ se usan como variables de entrada con las siguientes justificaciones físicas:

| Feature | Rol | Justificación física |
|---|---|---|
| `pm10_value` | Feature del día + lag t-1 | PM10 contiene PM2.5 como subconjunto — predictor colineal fuerte |
| `no2_value` | Feature del día + lag t-1 | Proxy de intensidad de tráfico y combustión — las mismas fuentes que emiten PM2.5 |
| `o3_value` | Feature del día + lag t-1 | **Anticorrelaciona** con PM2.5 (competencia fotoquímica). O₃ bajo en verano puede señalar aire estancado que acumula partículas |
| `so2_value` | Feature del día + lag t-1 | Indica actividad industrial que co-emite partículas finas |

Los valores del día actual no generan data leakage: al momento de la predicción (fin del día *t*), todas las lecturas de los cinco contaminantes para el día *t* ya están disponibles en la red de monitoreo.

---

### Decisiones de Diseño del Modelado

| Decisión | Elección | Justificación |
|---|---|---|
| Variable objetivo | Excedencia de PM2.5 (binario) | Mayor impacto sanitario; autocorrelación fuerte; umbral accionable |
| Horizontes | t+1 (24h) y t+2 (48h) | Operacionalmente útiles para avisos sanitarios y planificación industrial |
| Co-contaminantes | Features, no targets | NO₂, O₃, PM10, SO₂ codifican estado de emisiones y atmósfera |
| Métrica prioritaria | Recall (≥ 90%) | Perder un evento peligroso (falso negativo) es peor que una falsa alarma |
| Desbalanceo de clases | `scale_pos_weight` | Ajusta por la minoría de días peligrosos en el dataset |
| Estrategia de validación | Corte temporal estricto | Los splits aleatorios causan data leakage vía los lag features |
| Umbral de decisión | Calibrado por modelo | Se fija para alcanzar ≥ 90% de recall en validación, no en 0.5 |

---

### Outputs Esperados

| Output | Descripción |
|---|---|
| **Feature store procesado** | Parquet particionado por país/año — limpio, con join espacial aplicado |
| **Tablas de agregación** | Resúmenes por ciudad: días sobre umbrales UE/OMS, patrones estacionales |
| **Modelos ML entrenados** | LightGBM + XGBoost para t+1 y t+2, con umbrales calibrados para recall ≥ 90% |
| **Análisis SHAP** | Importancia de features por región y temporada |
| **API en producción** | FastAPI con endpoints `/predict`, `/predict/batch`, `/health`, `/metrics` |
| **Pipeline orquestado** | DAG de Prefect con schedule semanal e ingesta incremental |
| **Dashboard interactivo** | Plotly Dash: mapa europeo, series temporales, heatmap de estacionalidad, tabla de métricas |

---

### Stack Tecnológico

| Capa | Herramienta |
|---|---|
| Ingesta | Python, `airbase`, `requests` |
| Procesamiento | Apache Spark (PySpark) |
| Orquestación | Prefect |
| Modelado | LightGBM, XGBoost, scikit-learn, SHAP |
| Visualización | Plotly Dash |
| API | FastAPI |
| Containerización | Docker |
| CI/CD | GitHub Actions |
| Deploy | Render.com (free tier) |

---

### Fuentes de Datos

- **EEA Air Quality e-Reporting:** https://www.eea.europa.eu/en/datahub/datahubitem-view/778ef9f5-6293-4846-badd-56a29c70880d
- **NOAA GHCN Daily (bucket público AWS):** https://registry.opendata.aws/noaa-ghcn

---

*Project by — open to contributions and feedback.*
# air_pollution_forecasting_pipeline
