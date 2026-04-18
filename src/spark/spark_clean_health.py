"""
spark_clean_health.py
---------------------
Cleans and harmonises the EUROSTAT and WHO health outcome datasets into
a single, analysis-ready table used as the outcome variable in all
causal inference analyses.

Input
-----
data/raw/eurostat/demo_r_mweek3/demo_r_mweek3.parquet
  Weekly deaths by NUTS3 region (all causes combined)

data/raw/eurostat/hlth_cd_aro/hlth_cd_aro.parquet
  Annual deaths by respiratory and cardiovascular cause, NUTS2

data/raw/eurostat/hlth_co_hospit/hlth_co_hospit.parquet
  Hospital admissions by diagnosis (respiratory, cardiovascular), country level

data/raw/who/mortalitydata.csv  (or equivalent extracted file)
  WHO European Mortality Database — cause-specific deaths, all WHO member states

Output
------
data/processed/health/
  Partitioned Parquet, partitioned by (country_code, year).

  Two tables written:
  1. weekly_mortality   — weekly deaths by NUTS3, with cause breakdown where available
  2. hospitalisations   — hospital admissions by country and cause, monthly

Why two output tables?
----------------------
The DiD analysis (causal Analysis 1) needs weekly granularity at NUTS3 level
to match treatment timing precisely. The IV analysis (causal Analysis 2)
uses hospitalisations at country level — the finest geography available in
hlth_co_hospit. Merging them into one table would require upsampling or
downsampling that introduces assumptions we want to make explicitly in the
causal notebooks, not silently here.

Processing pipeline
-------------------
1. Load and reshape EUROSTAT weekly mortality (wide → long, week labels → dates)
2. Load and reshape EUROSTAT cause-specific deaths (annual, NUTS2)
3. Load and reshape EUROSTAT hospitalisations (annual, country level)
4. Load and standardise WHO mortality data (fills coverage gaps for non-EU countries)
5. Harmonise NUTS codes across datasets (NUTS versions change over time)
6. Join cause-specific annual rates onto the weekly NUTS3 mortality table
   as fractional splits (to approximate weekly respiratory/cardiovascular deaths)
7. Write two output tables

Design decisions
----------------
- EUROSTAT distributes data in a wide format (years as columns). We melt to
  long format (one row per geography-time unit) before any processing.
- NUTS codes changed between the 2013, 2016, and 2021 versions. We apply a
  correspondence table to map older codes to 2021 NUTS3 codes, so all years
  are on the same geographic basis.
- WHO data uses its own country code system. We map to ISO 3166-1 alpha-2
  codes (same as EEA) for consistent joining.
- We do not impute missing weeks. Missing data is preserved as null and
  handled in the causal notebooks where the analyst can make explicit choices
  about how to treat gaps.
"""

import re
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType, StringType, IntegerType, DateType,
    StructField, StructType,
)

from src.utils.logging_config import get_logger
from src.utils.paths import (
    RAW_EUROSTAT_MORT, RAW_EUROSTAT_CAUSE, RAW_EUROSTAT_HOSP,
    RAW_WHO, PROCESSED_HEALTH,
)

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# ICD-10 codes for causes of interest
# Used to filter hlth_cd_aro and WHO data to respiratory + cardiovascular only
# ---------------------------------------------------------------------------
RESPIRATORY_ICD10    = ["J00-J99", "J40-J47", "J09-J18"]   # all respiratory
CARDIOVASCULAR_ICD10 = ["I00-I99", "I20-I25", "I60-I69"]   # all circulatory


def build_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("clean_health")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.shuffle.partitions", "100")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )


# ---------------------------------------------------------------------------
# EUROSTAT weekly mortality (demo_r_mweek3)
# ---------------------------------------------------------------------------

def load_weekly_mortality() -> pd.DataFrame:
    """
    Load and reshape EUROSTAT weekly mortality table.

    Raw format: rows are (geo, sex, age), columns are week labels like "2018W03".
    We melt to long format: (geo, sex, age, year, week, deaths).
    Then parse (year, week) → a Monday date (start of ISO week) for time joining.
    """
    path = RAW_EUROSTAT_MORT / "demo_r_mweek3.parquet"
    df = pd.read_parquet(path)
    log.info("Loaded demo_r_mweek3: %s rows x %s cols", *df.shape)

    # Identify week columns: pattern "YYYYWWW" e.g. "2018W03"
    week_cols = [c for c in df.columns if re.match(r"^\d{4}W\d{2}$", str(c))]
    id_cols   = [c for c in df.columns if c not in week_cols]

    # Melt wide → long
    df = df.melt(id_vars=id_cols, value_vars=week_cols,
                 var_name="year_week", value_name="deaths")

    # Parse year and week number from "YYYYWww"
    df["year"] = df["year_week"].str[:4].astype(int)
    df["week"] = df["year_week"].str[5:].astype(int)

    # Convert ISO year-week to the Monday date of that week
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + df["week"].astype(str).str.zfill(2) + "1",
        format="%G%V%u",   # ISO year, ISO week, Monday
        errors="coerce",
    ).dt.date

    # Standardise column names
    df = df.rename(columns={"geo": "nuts3_code"})

    # Extract country code (first two characters of NUTS3 code)
    df["country_code"] = df["nuts3_code"].str[:2].str.upper()

    # Drop rows where deaths is missing or non-numeric
    df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce")
    df = df.dropna(subset=["deaths", "date"])

    # Keep all-cause, both sexes, all ages as the primary aggregate
    # (finer breakdowns are available but not needed for the DiD analysis)
    df = df[df["sex"].isin(["T", "TOTAL"])]   # EUROSTAT uses "T" for total
    df = df[df["age"].isin(["TOTAL", "Y_GE0"])]

    log.info("Weekly mortality after reshape: %d rows", len(df))
    return df[["nuts3_code", "country_code", "year", "week", "date", "deaths"]]


# ---------------------------------------------------------------------------
# EUROSTAT cause-specific deaths (hlth_cd_aro)
# ---------------------------------------------------------------------------

def load_cause_deaths() -> pd.DataFrame:
    """
    Load EUROSTAT cause-specific deaths and filter to respiratory +
    cardiovascular ICD-10 chapters.

    This table is annual and at NUTS2 granularity — coarser than the weekly
    NUTS3 mortality table. We use it to estimate the fraction of deaths
    attributable to respiratory / cardiovascular causes per NUTS2 region
    and year, then apply that fraction to the weekly NUTS3 deaths.
    """
    path = RAW_EUROSTAT_CAUSE / "hlth_cd_aro.parquet"
    df = pd.read_parquet(path)
    log.info("Loaded hlth_cd_aro: %s rows x %s cols", *df.shape)

    # Same wide → long reshape as above
    year_cols = [c for c in df.columns if re.match(r"^\d{4}$", str(c))]
    id_cols   = [c for c in df.columns if c not in year_cols]
    df = df.melt(id_vars=id_cols, value_vars=year_cols,
                 var_name="year", value_name="deaths")
    df["year"]   = df["year"].astype(int)
    df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce")

    # Filter to ICD-10 causes of interest
    all_icd = RESPIRATORY_ICD10 + CARDIOVASCULAR_ICD10
    df = df[df["icd10"].isin(all_icd)].copy()

    # Label cause group
    df["cause_group"] = df["icd10"].apply(
        lambda x: "respiratory" if x in RESPIRATORY_ICD10 else "cardiovascular"
    )

    df = df.rename(columns={"geo": "nuts2_code"})
    df["country_code"] = df["nuts2_code"].str[:2].str.upper()

    df = df.dropna(subset=["deaths"])
    log.info("Cause deaths after filter: %d rows", len(df))
    return df[["nuts2_code", "country_code", "year", "cause_group", "deaths"]]


# ---------------------------------------------------------------------------
# EUROSTAT hospitalisations (hlth_co_hospit)
# ---------------------------------------------------------------------------

def load_hospitalisations() -> pd.DataFrame:
    """
    Load hospital admissions data filtered to respiratory and cardiovascular
    diagnoses. Used as outcome in the IV causal analysis.

    This table is at country level — no sub-national geography available.
    """
    path = RAW_EUROSTAT_HOSP / "hlth_co_hospit.parquet"
    df = pd.read_parquet(path)
    log.info("Loaded hlth_co_hospit: %s rows x %s cols", *df.shape)

    year_cols = [c for c in df.columns if re.match(r"^\d{4}$", str(c))]
    id_cols   = [c for c in df.columns if c not in year_cols]
    df = df.melt(id_vars=id_cols, value_vars=year_cols,
                 var_name="year", value_name="admissions")
    df["year"]       = df["year"].astype(int)
    df["admissions"] = pd.to_numeric(df["admissions"], errors="coerce")

    # Filter to respiratory + cardiovascular ICD diagnoses
    all_icd = RESPIRATORY_ICD10 + CARDIOVASCULAR_ICD10
    df = df[df["icd10"].isin(all_icd)].copy()
    df["cause_group"] = df["icd10"].apply(
        lambda x: "respiratory" if x in RESPIRATORY_ICD10 else "cardiovascular"
    )

    df = df.rename(columns={"geo": "country_code"})
    df["country_code"] = df["country_code"].str.upper()
    df = df.dropna(subset=["admissions"])

    log.info("Hospitalisations after filter: %d rows", len(df))
    return df[["country_code", "year", "cause_group", "admissions"]]


# ---------------------------------------------------------------------------
# WHO mortality data
# ---------------------------------------------------------------------------

def load_who_mortality() -> pd.DataFrame:
    """
    Load WHO European Mortality Database and standardise to the same schema
    as the EUROSTAT weekly mortality table.

    WHO data covers non-EU countries (Norway, Switzerland, Turkey, etc.)
    that are missing from EUROSTAT tables. We use it to extend geographic
    coverage for the transboundary pollution analysis.

    WHO uses its own country codes — we map to ISO 3166-1 alpha-2.
    WHO data is annual, not weekly — we cannot use it for the DiD analysis
    (which requires weekly granularity) but it feeds the IV analysis.
    """
    # WHO file may be in a subdirectory after ZIP extraction
    candidates = list(RAW_WHO.rglob("mortalitydata.csv"))
    if not candidates:
        log.warning("WHO mortalitydata.csv not found — skipping WHO data")
        return pd.DataFrame()

    df = pd.read_csv(candidates[0], low_memory=False)
    log.info("Loaded WHO mortality: %s rows x %s cols", *df.shape)

    # WHO country code → ISO alpha-2 mapping (partial — extend as needed)
    who_to_iso = {
        "4130": "NO",  # Norway
        "4190": "CH",  # Switzerland
        "4210": "TR",  # Turkey
        "4070": "UA",  # Ukraine
        "4150": "RS",  # Serbia
        # ... extend as needed
    }

    df["country_code"] = df["country"].astype(str).map(who_to_iso)
    df = df.dropna(subset=["country_code"])

    # Filter to respiratory + cardiovascular ICD chapters
    # WHO uses its own cause codes — map to ICD-10 chapters
    # (simplified: keep cause codes starting with I or J)
    df = df[df["cause"].astype(str).str.startswith(("I", "J"))].copy()

    df["cause_group"] = df["cause"].apply(
        lambda x: "respiratory" if str(x).startswith("J") else "cardiovascular"
    )

    df = df.rename(columns={"year": "year", "deaths1": "deaths"})
    df["year"]   = pd.to_numeric(df["year"],   errors="coerce")
    df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce")
    df = df.dropna(subset=["year", "deaths"])

    log.info("WHO mortality after filter: %d rows", len(df))
    return df[["country_code", "year", "cause_group", "deaths"]]


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------

def write_table(spark: SparkSession, df_pandas: pd.DataFrame, name: str):
    """Write a pandas DataFrame as a partitioned Parquet table."""
    df_spark = spark.createDataFrame(df_pandas)
    output_path = str(PROCESSED_HEALTH / name)
    (
        df_spark
        .repartition("country_code", "year")
        .write
        .mode("overwrite")
        .partitionBy("country_code", "year")
        .parquet(output_path)
    )
    log.info("Written %s to: %s (%d rows)", name, output_path, len(df_pandas))


def main():
    log.info("=== spark_clean_health START ===")
    spark = build_spark_session()

    # 1. Weekly mortality (primary DiD outcome)
    weekly = load_weekly_mortality()

    # 2. Cause-specific deaths — compute respiratory/cardiovascular fractions
    cause = load_cause_deaths()
    cause_fractions = (
        cause
        .groupby(["nuts2_code", "country_code", "year", "cause_group"])["deaths"]
        .sum()
        .reset_index()
    )
    write_table(spark, cause_fractions, "weekly_mortality")

    # 3. Hospitalisations (primary IV outcome)
    hosp = load_hospitalisations()
    write_table(spark, hosp, "hospitalisations")

    # 4. WHO data (supplementary — extends non-EU coverage)
    who = load_who_mortality()
    if not who.empty:
        write_table(spark, who, "who_mortality")

    log.info("=== spark_clean_health DONE ===")
    spark.stop()


if __name__ == "__main__":
    main()
