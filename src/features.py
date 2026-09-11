import pandas as pd
import numpy as np


def add_time_features(df, timestamp_col):

    df[timestamp_col] = pd.to_datetime(
        df[timestamp_col]
    )

    df["hour"] = df[timestamp_col].dt.hour

    df["day"] = df[timestamp_col].dt.day

    df["month"] = df[timestamp_col].dt.month

    df["day_of_week"] = (
        df[timestamp_col].dt.dayofweek
    )

    return df