"""Cleaning, NaN handling, and feature pipeline."""

import numpy as np
import pandas as pd

from ..config import DROP_COLUMNS, TARGET_COLUMN

# Column-name aliases that may appear in different CSV versions.
# Maps raw column name -> canonical target name.
_TARGET_ALIASES = {
    "total_grid": "Energy Consumption",
}


def clean_and_prepare(df: pd.DataFrame, drop_columns=None, target_column=None) -> pd.DataFrame:
    """
    Remove redundant columns, move target to end, and optionally drop rows with NaNs.

    Also resolves known column-name aliases so that both `Data-Melbourne_F.csv`
    (which uses `total_grid`) and `Data-Melbourne_F_fixed.csv` (which uses
    `Energy Consumption`) work transparently.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.
    drop_columns : list, optional
        Columns to drop (default from config: e.g. 'Unnamed: 0').
    target_column : str, optional
        Target column name (default from config: 'Energy Consumption').

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with target as last column.

    Raises
    ------
    KeyError
        If the target column cannot be found even after alias resolution.
    """
    drop_columns = drop_columns or DROP_COLUMNS
    target_column = target_column or TARGET_COLUMN

    df_clean = df.drop(columns=[c for c in drop_columns if c in df.columns], errors="ignore")

    # Resolve aliases (e.g. 'total_grid' -> 'Energy Consumption')
    rename_map = {
        alias: canonical
        for alias, canonical in _TARGET_ALIASES.items()
        if alias in df_clean.columns and canonical not in df_clean.columns
    }
    if rename_map:
        df_clean = df_clean.rename(columns=rename_map)

    if target_column in df_clean.columns:
        target_series = df_clean.pop(target_column)
        df_clean[target_column] = target_series
    else:
        available = df_clean.columns.tolist()
        raise KeyError(
            f"Target column '{target_column}' not found after cleaning. "
            f"Available columns: {available}\n"
            "Set the correct column name via the target_column argument or TARGET_COLUMN in config.py."
        )

    # Drop rows with any NaN (original dataset had none, but guard for safety)
    # df_clean = df_clean.dropna()
    return df_clean


CALENDAR_FEATURES = ["t", "doy_sin", "doy_cos", "weekend"]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append deterministic calendar features derived from year/month/day.

    The energy series has a slow upward trend plus an annual cycle, which raw
    ``month`` / ``year`` / ``day`` integers expose poorly (December and January
    are far apart numerically). Added columns: ``t`` (days since first record),
    ``doy_sin`` / ``doy_cos`` (annual cycle) and ``weekend``. They depend only
    on the date, so they cannot leak the target. The target stays last.
    """
    out = df.copy()
    dt = pd.to_datetime(out[["year", "month", "day"]])
    doy = dt.dt.dayofyear.to_numpy()
    out["t"] = (dt - dt.min()).dt.days.to_numpy(dtype=float)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["weekend"] = (dt.dt.dayofweek >= 5).astype(float).to_numpy()
    target = out.pop(TARGET_COLUMN)
    out[TARGET_COLUMN] = target
    return out
