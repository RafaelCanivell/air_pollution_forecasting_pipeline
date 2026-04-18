"""
spark_clean_eea.py
------------------
PySpark job that cleans raw EEA Air Quality Parquet files and writes a
partitioned, analysis-ready dataset to data/processed/eea/.

Input
-----
data/raw/eea/<pollutant>_<verification>.parquet
  - Ten files: five pollutants × two verification levels (E1a, E2a)
  - Each file contains measurements from thousands of monitoring stations
    across all EU member states, one row per station-day

Output
------
data/processed/eea/
  Partitioned Parquet dataset, partitioned by (country_code, year).
  Partitioning by country + year makes downstream Spark joins efficient:
  the Spark join in spark_join_features.py can prune partitions and avoid
  reading the entire dataset for a single country or time window.

What this script does
---------------------
1. Load all ten raw Parquet files into a single Spark DataFrame
2. Rename and type-cast columns to a clean, consistent schema
3. Parse timestamps and extract date, year, month, day-of-week
4. Extract country code from the station EoI code (first two characters)
5. Handle missing values:
   - Rows with Validity < 0 (flagged invalid by EEA) are dropped
   - Rows with null Value are dropped
   - Rows with Value < 0 are dropped (sensor artefacts)
6. Clip extreme outliers using physically defensible upper bounds per pollutant
7. Deduplicate: keep one row per (station, pollutant, date) — E1a preferred
   over E2a when both exist for the same station-day (verified > unverified)
8. Pivot: one row per (station, date) with one column per pollutant
   This wide format is what spark_join_features.py expects
9. Write partitioned Parquet to data/processed/eea/

Design decisions
----------------
- We do NOT impute missing values here. Imputation decisions belong in
  spark_join_features.py where the full feature context is available.
- We keep both E1a and E2a rows during loading and deduplicate at step 7.
  This maximises temporal coverage: recent months only have E2a data.
- Upper bounds for outlier clipping are based on the highest credible
  recorded values in Europe — not arbitrary percentiles. We want to keep
  genuine extreme pollution events; only sensor spikes are removed.
- Spark is justified here: the combined raw EEA dataset across all countries
  and years runs to hundreds of millions of rows.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import DoubleType, DateType

from src.utils.logging_config import get_logger
from src.utils.paths import RAW_EEA, PROCESSED_EEA

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Physical upper bounds for outlier clipping (µg/m³)
# Source: EEA data quality guidelines + WHO/EU limit values + expert judgment
# ---------------------------------------------------------------------------
POLLUTANT_UPPER_BOUNDS = {
    "PM2.5": 900.0,   # highest credible daily mean in Europe (severe episode)
    "PM10":  1500.0,
    "NO2":   1000.0,
    "O3":    400.0,
    "SO2":   2000.0,
}

# EEA Validity codes: >= 1 means valid; negative codes mean invalid/dubious
MIN_VALIDITY = 1

# Column name mapping: raw EEA names → clean internal names
COLUMN_RENAME = {
    "AirQualityStationEoICode": "station_id",
    "AirPollutant":             "pollutant",
    "Start":                    "timestamp_start",
    "End":                      "timestamp_end",
    "Value":                    "value",
    "Validity":                 "validity",
    "Verification":             "verification",
}


def build_spark_session() -> SparkSession:
    """
    Initialise a SparkSession configured for local EEA processing.

    Memory settings are tuned for a machine with 16+ GB RAM.
    Adjust driver.memory if running on a smaller machine.
    """
    return (
        SparkSession.builder
        .appName("clean_eea")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.shuffle.partitions", "200")
        # Use Parquet page compression to reduce output size
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )


def load_raw_eea(spark: SparkSession):
    """
    Load all ten raw EEA Parquet files into one DataFrame.

    We add a 'verification_level' column (E1a / E2a) derived from the
    filename so we can prefer E1a rows during deduplication later.
    Spark reads all files in parallel — no loop needed.
    """
    from pyspark.sql.types import StringType

    dfs = []
    for parquet_path in sorted(RAW_EEA.glob("*.parquet")):
        # Extract verification level from filename: "PM2.5_E1a.parquet" → "E1a"
        verification_level = parquet_path.stem.split("_")[-1]

        df = spark.read.parquet(str(parquet_path))
        df = df.withColumn(
            "verification_level",
            F.lit(verification_level).cast(StringType())
        )
        dfs.append(df)
        log.info("Loaded: %s  (%s rows)", parquet_path.name, df.count())

    combined = dfs[0]
    for df in dfs[1:]:
        combined = combined.unionByName(df, allowMissingColumns=True)

    log.info("Combined EEA DataFrame: %s rows", combined.count())
    return combined


def rename_and_cast(df):
    """
    Rename columns to clean internal names and cast to appropriate types.

    Raw EEA Parquet has all columns as strings — explicit casting ensures
    Spark statistics and optimisations work correctly downstream.
    """
    for old_name, new_name in COLUMN_RENAME.items():
        if old_name in df.columns:
            df = df.withColumnRenamed(old_name, new_name)

    df = (
        df
        .withColumn("value",    F.col("value").cast(DoubleType()))
        .withColumn("validity", F.col("validity").cast("integer"))
    )
    return df


def parse_dates(df):
    """
    Parse timestamps and extract calendar features.

    Raw EEA timestamps are ISO8601 strings (e.g. "2018-03-15T00:00:00+01:00").
    We extract the date (day-level granularity) — our model operates on
    daily means, not hourly readings.
    """
    df = (
        df
        .withColumn("date",         F.to_date(F.col("timestamp_start")).cast(DateType()))
        .withColumn("year",         F.year("date").cast("integer"))
        .withColumn("month",        F.month("date").cast("integer"))
        .withColumn("day_of_week",  F.dayofweek("date").cast("integer"))
    )
    return df


def extract_country(df):
    """
    Extract the two-letter country code from the station EoI code.

    EoI codes follow the pattern: <CC><station_identifier>
    e.g. "DE0001A" → country_code = "DE"
    This is the primary geographic key used in all downstream joins.
    """
    df = df.withColumn(
        "country_code",
        F.upper(F.substring(F.col("station_id"), 1, 2))
    )
    return df


def filter_invalid(df):
    """
    Remove rows flagged as invalid by the EEA quality control system.

    EEA Validity codes:
      1  = valid
      2  = valid (but use with caution)
     -1  = invalid
     -99 = missing
    We keep only rows with validity >= MIN_VALIDITY (i.e. 1 or 2).
    We also drop null and negative values (sensor artefacts / calibration errors).
    """
    before = df.count()
    df = (
        df
        .filter(F.col("validity") >= MIN_VALIDITY)
        .filter(F.col("value").isNotNull())
        .filter(F.col("value") >= 0)
    )
    after = df.count()
    log.info("Validity filter: %s rows removed (%s remaining)", before - after, after)
    return df


def clip_outliers(df):
    """
    Clip physically implausible values to pollutant-specific upper bounds.

    We use hard physical upper bounds rather than statistical percentiles
    because we want to preserve genuine extreme pollution events (which are
    exactly what the model needs to predict) while removing sensor spikes
    that exceed any credible atmospheric concentration.
    """
    for pollutant, upper in POLLUTANT_UPPER_BOUNDS.items():
        df = df.withColumn(
            "value",
            F.when(
                (F.col("pollutant") == pollutant) & (F.col("value") > upper),
                F.lit(upper)
            ).otherwise(F.col("value"))
        )
    return df


def aggregate_to_daily(df):
    """
    Aggregate sub-daily readings to daily means per station and pollutant.

    EEA data contains hourly readings for most stations. Our model operates
    at daily granularity, so we take the daily mean. We also compute the
    daily max (useful for exceedance detection) and the reading count
    (useful as a data quality indicator — days with few readings are less reliable).
    """
    df = (
        df
        .groupBy("station_id", "country_code", "pollutant", "date", "year", "month", "day_of_week")
        .agg(
            F.mean("value").alias("value_mean"),
            F.max("value").alias("value_max"),
            F.count("value").alias("reading_count"),
            # Keep the verification_level of the best reading of the day
            F.min("verification_level").alias("verification_level"),  # E1a < E2a alphabetically
        )
    )
    return df


def deduplicate_e1a_over_e2a(df):
    """
    For each (station, pollutant, date), keep E1a if both E1a and E2a exist.

    When the pipeline runs incrementally, the same station-day may appear
    in both the verified (E1a) and unverified (E2a) files once EEA publishes
    the verified version. We prefer E1a (verified) over E2a (unverified).

    Window function approach: rank rows within each (station, pollutant, date)
    group by verification_level — E1a ranks first (lower string value).
    """
    window = Window.partitionBy("station_id", "pollutant", "date").orderBy(
        F.col("verification_level").asc()  # E1a < E2a — keeps E1a
    )
    df = (
        df
        .withColumn("rank", F.row_number().over(window))
        .filter(F.col("rank") == 1)
        .drop("rank")
    )
    return df


def pivot_pollutants(df):
    """
    Pivot from long format (one row per station-pollutant-date) to wide format
    (one row per station-date, one column per pollutant).

    Wide format is required by spark_join_features.py, which joins this
    dataset with ERA5 meteorological data and health outcome data on
    (station_id, date).

    Output columns:
      station_id, country_code, date, year, month, day_of_week,
      pm25_mean, pm25_max, pm10_mean, pm10_max, no2_mean, no2_max,
      o3_mean, o3_max, so2_mean, so2_max,
      pm25_readings, pm10_readings, ...
    """
    # Pivot mean values
    df_mean = (
        df
        .groupBy("station_id", "country_code", "date", "year", "month", "day_of_week")
        .pivot("pollutant", ["PM2.5", "PM10", "NO2", "O3", "SO2"])
        .agg(F.first("value_mean"))
    )

    # Rename pivoted columns to snake_case
    rename_mean = {
        "PM2.5": "pm25_mean", "PM10": "pm10_mean",
        "NO2":   "no2_mean",  "O3":   "o3_mean",  "SO2": "so2_mean",
    }
    for old, new in rename_mean.items():
        if old in df_mean.columns:
            df_mean = df_mean.withColumnRenamed(old, new)

    # Pivot max values and reading counts separately, then join
    df_max = (
        df
        .groupBy("station_id", "date")
        .pivot("pollutant", ["PM2.5", "PM10", "NO2", "O3", "SO2"])
        .agg(F.first("value_max"))
    )
    rename_max = {
        "PM2.5": "pm25_max", "PM10": "pm10_max",
        "NO2":   "no2_max",  "O3":   "o3_max",  "SO2": "so2_max",
    }
    for old, new in rename_max.items():
        if old in df_max.columns:
            df_max = df_max.withColumnRenamed(old, new)

    df_wide = df_mean.join(df_max, on=["station_id", "date"], how="left")
    return df_wide


def write_output(df):
    """
    Write the cleaned, wide-format EEA dataset as partitioned Parquet.

    Partitioned by (country_code, year) so downstream Spark jobs can
    read only the partitions they need — critical for performance when
    processing a single country or a short time window.

    overwrite mode is intentional: this job is designed to be re-run
    cleanly if the raw data changes (e.g. EEA republishes a corrected file).
    """
    output_path = str(PROCESSED_EEA)
    (
        df
        .repartition("country_code", "year")   # co-locate rows by partition key
        .write
        .mode("overwrite")
        .partitionBy("country_code", "year")
        .parquet(output_path)
    )
    log.info("Written cleaned EEA data to: %s", output_path)


def main():
    log.info("=== spark_clean_eea START ===")
    spark = build_spark_session()

    df = load_raw_eea(spark)
    df = rename_and_cast(df)
    df = parse_dates(df)
    df = extract_country(df)
    df = filter_invalid(df)
    df = clip_outliers(df)
    df = aggregate_to_daily(df)
    df = deduplicate_e1a_over_e2a(df)
    df = pivot_pollutants(df)
    write_output(df)

    log.info("=== spark_clean_eea DONE ===")
    spark.stop()


if __name__ == "__main__":
    main()
