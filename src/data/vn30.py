import sys

sys.path.append("/mnt/d/workspace/ai/rl4ast/")

import pandas as pd

from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.config import INDICATORS

import itertools

from configs.tickers import (
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)

df_vni = pd.read_json("data/raw/VNINDEX.json")

df_vni.rename(
    columns={
        "t": "date",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    },
    inplace=True,
)
# Format date from timestamp to string
df_vni["date"] = pd.to_datetime(df_vni["date"], unit="s").dt.strftime("%Y-%m-%d")
df_vni["tic"] = "VNINDEX"

print(df_vni.tail())
df_vni.sort_values(["date", "tic"], ignore_index=True, inplace=True)

na_dates = df_vni[df_vni.isna().any(axis=1)].date.unique()
df_vni = df_vni[~df_vni["date"].isin(na_dates)]
df_vni = df_vni.fillna(0)

vni = data_split(df_vni, TRADE_START_DATE, TRADE_END_DATE)

vni.to_csv("data/df_vni.csv")
