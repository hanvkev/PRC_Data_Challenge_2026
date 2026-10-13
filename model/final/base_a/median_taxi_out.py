"""Median taxi-out seconds for airport + stand + runway.

Copy this file to add a new dynamic feature; its filename is the feature name.
fit receives only the current fold's training rows, including the target.
transform receives no target for validation or later predictions.
For target aggregates, training=True must exclude each row's own target.
"""
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]
TARGET = "TAXITIME_SEC_mvt"


def fit(train):
    if len(train) < 2:
        raise ValueError("median_taxi_out needs at least two training rows.")
    return {
        "combination": train.groupby(REQUIRED_COLUMNS, dropna=False, observed=True)[TARGET].median(),
        "airport": train.groupby("ADEP_mvt", dropna=False, observed=True)[TARGET].median(),
        "global": float(train[TARGET].median()),
    }


def _leave_one_out_median(data, keys):
    # Sort once per grouping and select the two middle positions after removing
    # each row. This avoids a Python loop over millions of individual rows.
    values = data[TARGET].to_numpy(dtype=float)
    groups = (data.groupby(keys, dropna=False, observed=True).ngroup().to_numpy()
              if keys else np.zeros(len(data), dtype=int))
    order = np.lexsort((values, groups))
    counts = np.bincount(groups)
    starts = np.cumsum(counts) - counts
    ranks = np.empty(len(data), dtype=int)
    ranks[order] = np.arange(len(data)) - np.repeat(starts, counts)
    sizes = counts[groups]
    usable = sizes > 1
    lower = (sizes[usable] - 2) // 2
    upper = (sizes[usable] - 1) // 2
    offsets = starts[groups[usable]]
    sorted_values = values[order]
    result = np.full(len(data), np.nan)
    result[usable] = (
        sorted_values[offsets + lower + (ranks[usable] <= lower)]
        + sorted_values[offsets + upper + (ranks[usable] <= upper)]
    ) / 2
    return pd.Series(result, index=data.index)


def transform(data, state, training=False):
    if training:
        combination = _leave_one_out_median(data, REQUIRED_COLUMNS)
        airport = _leave_one_out_median(data, ["ADEP_mvt"])
        overall = _leave_one_out_median(data, [])
        return combination.fillna(airport).fillna(overall)

    keys = pd.MultiIndex.from_frame(data[REQUIRED_COLUMNS])
    combination = pd.Series(state["combination"].reindex(keys).to_numpy(), index=data.index)
    airport = data["ADEP_mvt"].map(state["airport"])
    return combination.fillna(airport).fillna(state["global"])
