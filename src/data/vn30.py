import sys
import os

sys.path.append("/mnt/d/workspace/ai/rl4ast/")

import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import json

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.config import INDICATORS
from finrl import config_tickers

import itertools

from configs.tickers import VN_HIGH_VOL_TICKER

TRAIN_START_DATE = "2012-04-01"
TRAIN_END_DATE = "2023-03-31"
TRADE_START_DATE = "2023-04-01"
TRADE_END_DATE = "2024-04-01"
df_raw = pd.DataFrame()

for ticker in VN_HIGH_VOL_TICKER:
    df_ticker = pd.read_json("data/raw/{0}.json".format(ticker))

    df_ticker.rename(
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
    df_ticker["date"] = pd.to_datetime(df_ticker["date"], unit="s").dt.strftime(
        "%Y-%m-%d"
    )
    df_ticker["tic"] = ticker

    df_raw = pd.concat([df_raw, df_ticker], ignore_index=True)

print(df_raw.tail())
df_raw.sort_values(["date", "tic"], ignore_index=True, inplace=True)
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_turbulence=True,
    user_defined_feature=False,
)

processed = fe.preprocess_data(df_raw)

print(processed.head())
list_ticker = processed["tic"].unique().tolist()
list_date = list(
    pd.date_range(processed["date"].min(), processed["date"].max()).astype(str)
)
combination = list(itertools.product(list_date, list_ticker))

processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
    processed, on=["date", "tic"], how="left"
)
processed_full = processed_full[processed_full["date"].isin(processed["date"])]
processed_full = processed_full.sort_values(["date", "tic"], ignore_index=True)

processed_full = processed_full.dropna(axis=0)
processed_full = processed_full.fillna(0)

print(processed_full.tail())

train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)

print(train.tail())
print(trade.tail())

print(len(train))
print(len(trade))

train.to_csv("data/train_data.csv")
trade.to_csv("data/trade_data.csv")
