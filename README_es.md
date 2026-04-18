# Pipeline de Calidad del Aire e Impacto en Salud Pública
### EEA · ERA5 · EUROSTAT — Francia, España, Bélgica, Países Bajos, Alemania (2019–2023)

> **Proyecto en curso.** Desarrollo activo — algunas fases y documentos están incompletos.

> **Filosofía de ingeniería.** Construido con un enfoque deliberado en principios MLOps: fiable, reutilizable, mantenible, flexible y completamente reproducible.

[![CI](https://github.com/your-org/eea-era5-air-quality/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/eea-era5-air-quality/actions)

---

## Qué hace este proyecto

Un pipeline end-to-end de ML e inferencia causal que:

1. **Predice** episodios peligrosos de excedencia de PM2.5 con 24–48 horas de antelación (LightGBM / XGBoost, recall ≥ 90 %)
2. **Estima** el efecto causal de esos episodios sobre la mortalidad respiratoria y las hospitalizaciones cardiovasculares (DiD, IV, Causal Forest)
3. **Sirve** ambos outputs a través de una API REST lista para producción

```
EEA + ERA5 + EUROSTAT
        │
        ▼
  Feature store (Parquet)
        │
   ┌────┴────┐
   │         │
   ▼         ▼
Flujo       Flujo
predictivo  causal
 (alerta)  (estimación de política)
   │         │
   └────┬────┘
        ▼
    API REST
 /predict · /health-impact · /metrics
```

---

## Quién debe leer qué

| Perfil | Por dónde empezar |
|---|---|
| ML / Ingeniero de datos | Este README → [docs/METHODOLOGY.md](docs/METHODOLOGY.md) → [docs/MONITORING.md](docs/MONITORING.md) |
| Epidemiólogo / Investigador | [docs/CAUSAL_INFERENCE.md](docs/CAUSAL_INFERENCE.md) → [docs/CAUSAL_DAG.md](docs/CAUSAL_DAG.md) |
| Analista de políticas | [docs/TARGET_AUDIENCE.md](docs/TARGET_AUDIENCE.md) → Endpoints de API abajo |
| Nuevo contribuidor | [docs/METHODOLOGY.md](docs/METHODOLOGY.md) → [docs/DAG_PREFECT.md](docs/DAG_PREFECT.md) |

---

## Prerrequisitos

- Python ≥ 3.10, Java 11+ (para Spark), Docker
- [Cuenta en Copernicus CDS](https://cds.climate.copernicus.eu) — gratuita, necesaria para ERA5
- ~25 GB de disco para datos crudos y procesados (5 países, 5 años)

---

## Inicio rápido

```bash
# 1. Clonar e instalar dependencias de ingesta
git clone https://github.com/your-org/eea-era5-air-quality.git
pip install -r requirements-ingestion.txt

# 2. Configurar acceso a ERA5 (una sola vez)
# Regístrate en https://cds.climate.copernicus.eu -> guarda la clave en ~/.cdsapirc

# 3. Dry-run para comprobar tamaños de descarga
python src/ingestion/download_eea.py --dry-run
python src/ingestion/download_era5.py --dry-run

# 4. Pipeline completo via Makefile
make data      # descarga + validación + procesamiento Spark
make train     # ingeniería de features + entrenamiento
make causal    # notebooks DiD, IV, Causal Forest
make serve     # lanza FastAPI en localhost:8000
```

Modo local (sin Spark):
```bash
spark-submit src/spark/spark_clean_eea.py --engine pandas
```

---

## Endpoints de la API

URL base: `https://your-app.onrender.com` · [Swagger UI](http://localhost:8000/docs)

| Endpoint | Método | Descripción |
|---|---|---|
| `/predict` | POST | Probabilidad de excedencia de PM2.5 a t+1 y t+2 |
| `/predict/batch` | POST | Lo mismo para múltiples estaciones |
| `/health-impact` | GET | Estimación causal de muertes / ingresos adicionales |
| `/metrics` | GET | Recall, precisión y scores de drift en tiempo real |
| `/health` | GET | Health check del servicio |

**Ejemplo de petición a `/predict`:**
```json
{
  "station_id": "FR04012",
  "timestamp": "2024-01-15",
  "features": {
    "pm10_value": 45.2, "no2_value": 38.1, "o3_value": 52.3,
    "so2_value": 3.4, "temperature": 2.5, "wind_speed": 4.1,
    "boundary_layer_height": 320
  }
}
```

---

## Documentación

| Documento | Contenido |
|---|---|
| [TARGET_AUDIENCE.md](docs/TARGET_AUDIENCE.md) | Para quién es este proyecto y cómo debe usarlo cada perfil |
| [METHODOLOGY.md](docs/METHODOLOGY.md) | Fases 1–6, decisiones de diseño, modelado, stack tecnológico |
| [CAUSAL_INFERENCE.md](docs/CAUSAL_INFERENCE.md) | DiD, IV, Causal Forest — métodos, supuestos, validaciones |
| [CAUSAL_DAG.md](docs/CAUSAL_DAG.md) | Grafo acíclico dirigido — relaciones entre variables, caminos bloqueados |
| [MONITORING.md](docs/MONITORING.md) | Métricas offline, detección de drift en producción, lógica de reentrenamiento |
| [DAG_PREFECT.md](docs/DAG_PREFECT.md) | Orquestación con Prefect — grafo de tareas, flag gates, programación |
| [LIMITATIONS.md](docs/LIMITATIONS.md) | Limitaciones conocidas — leer antes de interpretar resultados |

---

## Fuentes de datos

| Fuente | Dataset |
|---|---|
| EEA | [Air Quality e-Reporting](https://www.eea.europa.eu/en/datahub/datahubitem-view/778ef9f5-6293-4846-badd-56a29c70880d) |
| Copernicus/ECMWF | [ERA5 Reanalysis](https://cds.climate.copernicus.eu) |
| EUROSTAT | `demo_r_mweek3`, `hlth_cd_aro`, `hlth_co_hospit` |
| WHO Europa | [European Mortality Database](https://gateway.euro.who.int/en/datasets/european-mortality-database) |

---

*Abierto a contribuciones y feedback.*
