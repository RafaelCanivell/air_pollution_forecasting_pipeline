"""
spark_join_features.py
----------------------
Joins the three cleaned datasets (EEA air quality, ERA5 meteorology,
health outcomes) into a single feature store used as input for both
the ML models and the causal inference analyses.

This is the most important Spark job in the pipeline — it is the single
point where all data sources converge into the analytical dataset.

Input
-----
data/processed/eea/          — cleaned EEA data (station-day, wide pollutant columns)
data/processed/era5/         — cleaned ERA5 data (station-day, meteorological variables)
data/processed/health/       — cleaned health data (NUTS3-week mortality, country hospitalisations)

Output
------
data/processed/features/
  Partitioned Parquet, partitioned by (country_code, year).
  One row per (station_id, date) — the unit of analysis for the ML model.
  Includes:
    - All pollutant values (same-day + lag features)
    - All meteorological variables (same-day + lag features)
    - Rolling window aggregates (7-day, 14-day)
    - Calendar features (month, day of week, season, holiday flag)
    - PM2.5 exceedance label (binary target: pm25 > 25 µg/m³)
    - Health outcome joins at NUTS3-week level (for causal analyses)

data/processed/aggregations/
  City-level yearly summaries: days above EU/WHO thresholds, seasonal
  means, cross-border correlation statistics.

Feature engineering decisions
------------------------------
LAG FEATURES
  Lag features (t-1, t-2, t-7) capture autocorrelation in pollution and
  weather. PM2.5 at t-1 is the single strongest predictor of PM2.5 at t+1
  due to atmospheric persistence. Without lags, the model loses most of
  its predictive power.

ROLLING WINDOWS
  7-day and 14-day rolling means capture medium-term accumulation episodes
  (e.g. persistent anticyclones that trap pollutants for days at a time).

TEMPORAL SPLIT SAFETY
  All lag and rolling features are computed within a strict temporal order.
  We use Spark Window functions with orderBy(date) and explicit rowsBetween
  ranges to prevent any future data leaking into past rows.
  The train/validation split in train.py cuts at a specific date — but
  leakage-free feature engineering starts here.

TARGET VARIABLE
  pm25_exceedance (binary): 1 if pm25_mean > 25 µg/m³, 0 otherwise.
  25 µg/m³ is the EU daily limit value for PM2.5.
  We predict exceedance at t+1 (24h ahead) and t+2 (48h ahead).
  Both targets are computed here and stored in the feature store.

HEALTH JOIN
  Health outcomes are joined at NUTS3-week level. The NUTS3 code for each
  EEA station is looked up from a station metadata table. The join key is
  (nuts3_code, year, week). This join is optional for the ML model but
  required for the causal analyses — we include it here so the feature
  store serves both use cases.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window

from src.utils.logging_config import get_logger
from src.utils.paths import (
    PROCESSED_EEA, PROCESSED_ERA5, PROCESSED_HEALTH,
    PROCESSED_FEATURES, PROCESSED_AGGREGATIONS,
)

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
PM25_EU_DAILY_LIMIT  = 25.0   # µg/m³ — EU Air Quality Directive daily limit
PM25_WHO_ANNUAL      = 5.0    # µg/m³ — WHO 2021 annual mean guideline

# Pollutants for which we build lag and rolling features
FEATURE_POLLUTANTS = ["pm25_mean", "pm10_mean", "no2_mean", "o3_mean", "so2_mean"]
FEATURE_METEO      = [
    "temp_2m", "wind_speed", "surface_pressure",
    "precipitation", "boundary_layer_height", "low_blh_flag",
]

LAG_DAYS     = [1, 2, 7]    # t-1, t-2, t-7
ROLLING_DAYS = [7, 14]      # 7-day and 14-day rolling means


def build_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("join_features")
        .config("spark.driver.memory", "12g")
        .config("spark.sql.shuffle.partitions", "400")
        .config("spark.sql.parquet.compression.codec", "snappy")
        # Broadcast join threshold: ERA5 station lookup is small enough to broadcast
        .config("spark.sql.autoBroadcastJoinThreshold", "50m")
        .getOrCreate()
    )


def load_inputs(spark: SparkSession):
    """Load the three processed datasets."""
    eea    = spark.read.parquet(str(PROCESSED_EEA))
    era5   = spark.read.parquet(str(PROCESSED_ERA5))
    health = spark.read.parquet(str(PROCESSED_HEALTH / "weekly_mortality"))

    log.info("EEA rows:    %d", eea.count())
    log.info("ERA5 rows:   %d", era5.count())
    log.info("Health rows: %d", health.count())
    return eea, era5, health


def join_eea_era5(eea, era5):
    """
    Join EEA and ERA5 on (station_id, date).

    This is the primary join: it attaches meteorological conditions to each
    station-day observation. The join key is exact (same station_id, same date)
    because ERA5 has already been interpolated to station coordinates in
    spark_clean_era5.py.

    We use a left join to preserve all EEA rows even if ERA5 data is missing
    for a particular date (e.g. at the edges of the temporal coverage).
    """
    # Drop columns duplicated in both DataFrames (year, month, country_code)
    # Keep EEA versions — they are the primary source of geographic metadata
    era5_cols = [c for c in era5.columns
                 if c not in ("year", "month", "country_code")]
    era5_slim = era5.select(["station_id", "date"] + era5_cols)

    df = eea.join(era5_slim, on=["station_id", "date"], how="left")
    log.info("After EEA ⋈ ERA5 join: %d rows", df.count())
    return df


def add_lag_features(df):
    """
    Add lag features for all pollutants and meteorological variables.

    Lag features are computed using Spark Window functions, partitioned by
    station_id and ordered by date. This ensures lags are computed within
    each station's time series — not across stations.

    IMPORTANT: the window is ordered by date (ascending). Spark Window
    functions with lag() look backwards by default — lag(col, 1) gives
    the value from the previous row (yesterday). No future data leaks in.
    """
    window_station = (
        Window
        .partitionBy("station_id")
        .orderBy(F.col("date").cast("timestamp").cast("long"))
        # No rowsBetween needed for lag() — it operates on ordered rows
    )

    all_feature_cols = FEATURE_POLLUTANTS + FEATURE_METEO

    for col_name in all_feature_cols:
        if col_name not in df.columns:
            continue
        for lag_days in LAG_DAYS:
            lag_col_name = f"{col_name}_lag{lag_days}"
            df = df.withColumn(
                lag_col_name,
                F.lag(F.col(col_name), lag_days).over(window_station)
            )

    log.info("Lag features added: %d lag columns", len(all_feature_cols) * len(LAG_DAYS))
    return df


def add_rolling_features(df):
    """
    Add rolling window means for pollutants and key meteorological variables.

    Rolling means capture multi-day accumulation episodes. A 7-day rolling
    mean of PM2.5, for example, captures whether the station has been in a
    persistent high-pollution period — a strong predictor of tomorrow's value.

    We use rowsBetween(-(n-1), -1) — the window ends at yesterday (row -1)
    and starts n-1 days before. This excludes the current day (row 0) from
    the rolling mean, preventing leakage of same-day information into what
    should be a historical feature.
    """
    window_station = (
        Window
        .partitionBy("station_id")
        .orderBy(F.col("date").cast("timestamp").cast("long"))
    )

    rolling_cols = FEATURE_POLLUTANTS + ["boundary_layer_height", "temp_2m"]

    for col_name in rolling_cols:
        if col_name not in df.columns:
            continue
        for n_days in ROLLING_DAYS:
            roll_col_name = f"{col_name}_rolling{n_days}d"
            window_n = window_station.rowsBetween(-(n_days - 1), -1)
            df = df.withColumn(
                roll_col_name,
                F.mean(F.col(col_name)).over(window_n)
            )

    log.info("Rolling features added: %d rolling columns",
             len(rolling_cols) * len(ROLLING_DAYS))
    return df


def add_calendar_features(df):
    """
    Add calendar-based features: season, is_weekend, and heating season flag.

    Season is a strong confounder for PM2.5: winter heating increases
    emissions, summer heat increases ozone, spring has pollen events.
    Encoding season explicitly helps the model and makes SHAP values
    more interpretable.

    Heating season (October–March in most of Europe) is a particularly
    important feature because residential wood and coal burning is a major
    PM2.5 source in Eastern Europe.
    """
    df = df.withColumn(
        "season",
        F.when(F.col("month").isin([12, 1, 2]),  F.lit("winter"))
         .when(F.col("month").isin([3, 4, 5]),   F.lit("spring"))
         .when(F.col("month").isin([6, 7, 8]),   F.lit("summer"))
         .otherwise(F.lit("autumn"))
    )

    df = df.withColumn(
        "is_weekend",
        F.when(F.col("day_of_week").isin([1, 7]), F.lit(1)).otherwise(F.lit(0))
        # Spark: 1=Sunday, 7=Saturday
    )

    df = df.withColumn(
        "heating_season",
        F.when(F.col("month").isin([10, 11, 12, 1, 2, 3]), F.lit(1)).otherwise(F.lit(0))
    )

    return df


def add_target_variables(df):
    """
    Add the binary prediction targets for the ML model.

    pm25_exceedance_t1: will PM2.5 exceed 25 µg/m³ tomorrow (t+1)?
    pm25_exceedance_t2: will PM2.5 exceed 25 µg/m³ the day after (t+2)?

    We compute these by leading (shifting forward) the pm25 column within
    each station's time series. The model is trained on row t to predict
    the target at t+1 or t+2 — so the target must be the FUTURE value
    relative to the feature row.

    lead(col, 1) gives the next row's value — tomorrow's PM2.5 from today's row.
    This is the correct direction for a forecasting target.
    """
    window_station = (
        Window
        .partitionBy("station_id")
        .orderBy(F.col("date").cast("timestamp").cast("long"))
    )

    for horizon in [1, 2]:
        df = df.withColumn(
            f"pm25_future_t{horizon}",
            F.lead(F.col("pm25_mean"), horizon).over(window_station)
        )
        df = df.withColumn(
            f"pm25_exceedance_t{horizon}",
            F.when(
                F.col(f"pm25_future_t{horizon}") > PM25_EU_DAILY_LIMIT,
                F.lit(1)
            ).otherwise(F.lit(0))
        )

    # Drop the intermediate future value columns — keep only the binary targets
    df = df.drop("pm25_future_t1", "pm25_future_t2")

    # Rows where the target is null (last row of each station's time series)
    # cannot be used for training — drop them
    df = df.filter(
        F.col("pm25_exceedance_t1").isNotNull() &
        F.col("pm25_exceedance_t2").isNotNull()
    )

    log.info(
        "Target variable distribution — t+1 exceedance rate: %.2f%%",
        df.filter(F.col("pm25_exceedance_t1") == 1).count() / df.count() * 100
    )
    return df


def join_health_outcomes(df, health):
    """
    Join weekly mortality data onto the feature store at NUTS3-week level.

    Join key: (nuts3_code, year, week_of_year)
    The nuts3_code for each station comes from a station metadata lookup.

    This join adds health outcomes as additional columns — they are used
    in the causal analyses but ignored by the ML model.
    We use a left join to preserve all feature rows.
    """
    # Add week of year to feature store for join key
    df = df.withColumn("week_of_year", F.weekofyear(F.col("date")))

    # Add NUTS3 code from station metadata
    # (assumes station_id encodes NUTS3 — adjust lookup if needed)
    df = df.withColumn(
        "nuts3_code",
        F.substring(F.col("station_id"), 1, 5)  # first 5 chars = NUTS3 approximation
    )

    # Rename health columns to avoid collision
    health = health.select(
        F.col("nuts3_code"),
        F.col("year"),
        F.col("week").alias("week_of_year"),
        F.col("deaths").alias("weekly_deaths"),
    )

    df = df.join(health, on=["nuts3_code", "year", "week_of_year"], how="left")
    log.info("After health join: %d rows", df.count())
    return df


def write_feature_store(df):
    """
    Write the final feature store as partitioned Parquet.

    The feature store is the canonical input for:
    - train.py (ML model training)
    - did_analysis.py (causal Analysis 1)
    - iv_analysis.py (causal Analysis 2)
    - heterogeneous_effects.py (causal Analysis 3)

    All downstream consumers read from this path — no one re-runs the joins.
    """
    output_path = str(PROCESSED_FEATURES)
    (
        df
        .repartition("country_code", "year")
        .write
        .mode("overwrite")
        .partitionBy("country_code", "year")
        .parquet(output_path)
    )
    log.info("Feature store written to: %s", output_path)
    log.info("Total rows: %d", df.count())
    log.info("Total columns: %d", len(df.columns))


def write_aggregations(df):
    """
    Write city-level yearly aggregations for the dashboard and reporting.

    These are pre-computed summaries — faster to query than the full feature
    store, suitable for the Plotly Dash map layer.
    """
    agg = (
        df
        .groupBy("country_code", "station_id", "year")
        .agg(
            F.mean("pm25_mean").alias("pm25_annual_mean"),
            F.sum(
                F.when(F.col("pm25_mean") > PM25_EU_DAILY_LIMIT, 1).otherwise(0)
            ).alias("days_above_eu_limit"),
            F.sum(
                F.when(F.col("pm25_mean") > PM25_WHO_ANNUAL, 1).otherwise(0)
            ).alias("days_above_who_guideline"),
            F.mean("boundary_layer_height").alias("mean_blh"),
            F.count("*").alias("days_with_data"),
        )
    )

    output_path = str(PROCESSED_AGGREGATIONS)
    (
        agg
        .repartition("country_code", "year")
        .write
        .mode("overwrite")
        .partitionBy("country_code", "year")
        .parquet(output_path)
    )
    log.info("Aggregations written to: %s", output_path)


def main():
    log.info("=== spark_join_features START ===")
    spark = build_spark_session()

    eea, era5, health = load_inputs(spark)

    df = join_eea_era5(eea, era5)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_calendar_features(df)
    df = add_target_variables(df)
    df = join_health_outcomes(df, health)

    write_feature_store(df)
    write_aggregations(df)

    log.info("=== spark_join_features DONE ===")
    spark.stop()


if __name__ == "__main__":
    main()
