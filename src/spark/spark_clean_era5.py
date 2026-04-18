"""
spark_clean_era5.py
-------------------
Converts raw ERA5 NetCDF files to Parquet and spatially interpolates the
ERA5 grid to the coordinates of EEA monitoring stations.

Why spatial interpolation is necessary
---------------------------------------
ERA5 provides data on a regular 0.25° × 0.25° grid (~28 km resolution).
EEA monitoring stations are point locations that rarely fall exactly on a
grid node. To assign meteorological conditions to each station, we
interpolate the four nearest ERA5 grid nodes to the station's coordinates
using bilinear interpolation — the standard approach in atmospheric science.

This replaces the simpler (and less accurate) nearest-neighbour approach
that would have been used with NOAA point station data.

Input
-----
data/raw/era5/era5_<YYYY>_<MM>.nc
  - One NetCDF file per (year, month), covering Europe (lon -25→45, lat 34→72)
  - Variables: t2m, u10, v10, sp, tp, blh (see download_era5.py)
  - Hourly resolution

data/processed/eea/
  - Cleaned EEA data (output of spark_clean_eea.py)
  - Used to extract the list of unique station coordinates for interpolation

Output
------
data/processed/era5/
  Partitioned Parquet, partitioned by (country_code, year).
  One row per (station_id, date) with daily aggregates of all six variables.
  This is the same granularity as the cleaned EEA data, ready for joining.

Processing pipeline
-------------------
1. Extract unique station locations from processed EEA data
2. For each NetCDF file:
   a. Open with xarray and aggregate hourly → daily means (and daily max for BLH)
   b. For each station, bilinearly interpolate the ERA5 grid to station coords
   c. Convert to pandas DataFrame
3. Combine all monthly DataFrames and convert to Spark
4. Add derived meteorological features (wind speed, temperature delta)
5. Write partitioned Parquet

Design decisions
----------------
- xarray handles the NetCDF → array conversion; Spark handles the
  distribution and writing. We don't try to read NetCDF directly in Spark
  (no native support) — instead we process each monthly file with xarray
  in a Spark UDF mapped over the list of files.
- Bilinear interpolation is done with xarray's .interp() method, which
  uses scipy under the hood. This is accurate and vectorised over all
  stations at once.
- We aggregate to daily BEFORE interpolating spatially — cheaper than
  interpolating 24 hourly grids per day.
- BLH (boundary layer height) is our causal instrument for the IV analysis.
  We keep both the daily mean and the daily minimum — low daily minimum BLH
  indicates a persistent inversion, which is the physically relevant signal.
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType, StringType, DateType, IntegerType,
    StructField, StructType,
)

from src.utils.logging_config import get_logger
from src.utils.paths import RAW_ERA5, PROCESSED_ERA5, PROCESSED_EEA

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Variable name mapping: ERA5 short names → clean internal names
# ---------------------------------------------------------------------------
ERA5_VAR_RENAME = {
    "t2m": "temp_2m",          # 2m temperature (K → converted to °C below)
    "u10": "wind_u10",         # 10m eastward wind component (m/s)
    "v10": "wind_v10",         # 10m northward wind component (m/s)
    "sp":  "surface_pressure", # surface pressure (Pa → hPa below)
    "tp":  "precipitation",    # total precipitation (m → mm below)
    "blh": "boundary_layer_height",  # BLH (m) — causal instrument
}


def build_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("clean_era5")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )


def get_station_locations(spark: SparkSession) -> pd.DataFrame:
    """
    Extract unique (station_id, latitude, longitude, country_code) from the
    cleaned EEA data.

    ERA5 interpolation needs station coordinates. Rather than maintaining a
    separate station metadata file, we derive them from the processed EEA
    data which already has station_id and country_code.

    Note: EEA raw data includes station lat/lon in the metadata columns.
    If those columns were not retained in spark_clean_eea.py, add them there.
    """
    df = spark.read.parquet(str(PROCESSED_EEA))

    stations = (
        df
        .select("station_id", "country_code", "latitude", "longitude")
        .dropDuplicates(["station_id"])
        .toPandas()
    )
    log.info("Found %d unique EEA stations for ERA5 interpolation", len(stations))
    return stations


def aggregate_hourly_to_daily(ds: xr.Dataset) -> xr.Dataset:
    """
    Aggregate an hourly ERA5 xarray Dataset to daily resolution.

    Aggregation rules:
    - temperature, wind, pressure, BLH → daily mean
    - precipitation → daily sum (it is an accumulated variable in ERA5)
    - BLH → also keep daily minimum (persistent inversions = low BLH all day)

    We resample to calendar days in UTC. ERA5 timestamps are UTC.
    """
    daily_mean = ds[["t2m", "u10", "v10", "sp", "blh"]].resample(time="1D").mean()
    daily_sum  = ds[["tp"]].resample(time="1D").sum()
    daily_min  = ds[["blh"]].resample(time="1D").min().rename({"blh": "blh_min"})

    daily = xr.merge([daily_mean, daily_sum, daily_min])
    return daily


def unit_conversions(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert ERA5 variables from native units to more interpretable units.

    ERA5 native units → converted units:
    - t2m:  Kelvin     → Celsius    (subtract 273.15)
    - sp:   Pascals    → hPa        (divide by 100)
    - tp:   metres     → mm         (multiply by 1000)
    - blh:  metres     → metres     (no conversion, already interpretable)
    - wind: m/s        → m/s        (no conversion)
    """
    ds["t2m"] = ds["t2m"] - 273.15
    ds["sp"]  = ds["sp"]  / 100.0
    ds["tp"]  = ds["tp"]  * 1000.0
    return ds


def interpolate_to_stations(
    daily_ds: xr.Dataset,
    stations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Bilinearly interpolate ERA5 daily grid to EEA station coordinates.

    xarray's .interp() method performs bilinear interpolation when given
    arrays of target coordinates. We pass all station latitudes and
    longitudes at once — this is vectorised over stations, so it's fast
    even for thousands of stations.

    Returns a long-format pandas DataFrame:
      (station_id, date, t2m, u10, v10, sp, tp, blh, blh_min)
    """
    lats = xr.DataArray(stations["latitude"].values,  dims="station")
    lons = xr.DataArray(stations["longitude"].values, dims="station")

    # Interpolate all variables to all station locations simultaneously
    interpolated = daily_ds.interp(
        latitude=lats,
        longitude=lons,
        method="linear",
    )

    # Convert xarray Dataset → pandas DataFrame
    # Result shape: (n_days × n_stations) rows
    df = interpolated.to_dataframe().reset_index()

    # Add station identifiers (the 'station' dimension is just an integer index)
    station_map = stations.reset_index(drop=True)
    df["station_id"]   = df["station"].map(station_map["station_id"])
    df["country_code"] = df["station"].map(station_map["country_code"])
    df = df.drop(columns=["station", "latitude", "longitude"])

    # Rename ERA5 variables to clean internal names
    df = df.rename(columns={**ERA5_VAR_RENAME, "blh_min": "boundary_layer_height_min"})

    # Rename time column to date
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived meteorological features that are more informative
    for PM2.5 prediction than the raw ERA5 variables alone.

    wind_speed: scalar wind speed from u and v components.
      More interpretable than directional components separately.
      High wind speed → pollutant dispersion; low wind → accumulation.

    temp_inversion_proxy: difference between surface temperature and
      boundary layer temperature. A positive value suggests the air
      above the surface is warmer than the surface — an inversion —
      which traps pollutants near the ground.
      Approximated here as -BLH (lower BLH = stronger inversion signal).
      The IV analysis uses BLH directly as the causal instrument.
    """
    df["wind_speed"] = np.sqrt(df["wind_u10"] ** 2 + df["wind_v10"] ** 2)

    # Low BLH → strong inversion → high PM2.5 accumulation risk
    # We keep the raw BLH as the IV instrument and add this derived flag
    df["low_blh_flag"] = (df["boundary_layer_height"] < 500).astype(int)

    return df


def process_one_month(
    nc_path: Path,
    stations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full processing pipeline for a single ERA5 monthly NetCDF file.

    This function is designed to be called either directly or inside a
    Spark mapPartitions UDF. Each monthly file is independent, so
    parallelisation over months is straightforward.
    """
    log.info("Processing: %s", nc_path.name)

    ds = xr.open_dataset(str(nc_path))
    ds = aggregate_hourly_to_daily(ds)
    ds = unit_conversions(ds)

    df = interpolate_to_stations(ds, stations)
    df = add_derived_features(df)

    ds.close()

    # Add year and month for partitioning
    df["date"] = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["date"]  = df["date"].dt.date

    log.info("Processed %s: %d rows", nc_path.name, len(df))
    return df


def write_output(spark: SparkSession, df_pandas: pd.DataFrame):
    """
    Convert the combined pandas DataFrame to a Spark DataFrame and write
    as partitioned Parquet.

    We process all months in pandas (manageable: ~300 MB total for one year
    after interpolation to station level) and convert to Spark only for the
    final write — this avoids the overhead of serialising xarray objects
    across Spark executors.
    """
    schema = StructType([
        StructField("station_id",                  StringType(),  True),
        StructField("country_code",                StringType(),  True),
        StructField("date",                        DateType(),    True),
        StructField("year",                        IntegerType(), True),
        StructField("month",                       IntegerType(), True),
        StructField("temp_2m",                     DoubleType(),  True),
        StructField("wind_u10",                    DoubleType(),  True),
        StructField("wind_v10",                    DoubleType(),  True),
        StructField("wind_speed",                  DoubleType(),  True),
        StructField("surface_pressure",            DoubleType(),  True),
        StructField("precipitation",               DoubleType(),  True),
        StructField("boundary_layer_height",       DoubleType(),  True),
        StructField("boundary_layer_height_min",   DoubleType(),  True),
        StructField("low_blh_flag",                IntegerType(), True),
    ])

    df_spark = spark.createDataFrame(df_pandas, schema=schema)

    output_path = str(PROCESSED_ERA5)
    (
        df_spark
        .repartition("country_code", "year")
        .write
        .mode("overwrite")
        .partitionBy("country_code", "year")
        .parquet(output_path)
    )
    log.info("Written cleaned ERA5 data to: %s", output_path)


def main():
    log.info("=== spark_clean_era5 START ===")
    spark = build_spark_session()

    # Step 1: get station locations from already-processed EEA data
    stations = get_station_locations(spark)

    # Step 2: process each monthly NetCDF file sequentially
    # (parallelisation over months would require distributing xarray objects
    # across executors — complex; sequential is fast enough for 12×N files)
    nc_files = sorted(RAW_ERA5.glob("era5_*.nc"))
    if not nc_files:
        log.error("No ERA5 NetCDF files found in %s", RAW_ERA5)
        return

    all_dfs = []
    for nc_path in nc_files:
        try:
            df_month = process_one_month(nc_path, stations)
            all_dfs.append(df_month)
        except Exception as exc:
            log.error("Failed to process %s: %s", nc_path.name, exc)

    if not all_dfs:
        log.error("No ERA5 files processed successfully — aborting")
        return

    # Step 3: combine all months and write
    df_combined = pd.concat(all_dfs, ignore_index=True)
    log.info("Combined ERA5 DataFrame: %d rows", len(df_combined))

    write_output(spark, df_combined)

    log.info("=== spark_clean_era5 DONE ===")
    spark.stop()


if __name__ == "__main__":
    main()
