import pandas as pd
import numpy as np
import datetime
import yfinance as yf

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import INDICATORS

import itertools

print("Downloading data from Yahoo Finance")
# aapl_df_yf = yf.download(tickers="aapl", start="2020-01-01", end="2020-01-31")
# aapl_df_yf.head()

# aapl_df_finrl = YahooDownloader(
#     start_date="2020-01-01", end_date="2020-01-31", ticker_list=["aapl"]
# ).fetch_data()
# aapl_df_finrl.head()

print(config_tickers.DOW_30_TICKER)

TRAIN_START_DATE = "2011-01-01"
TRAIN_END_DATE = "2022-01-01"
TRADE_START_DATE = "2022-01-02"
TRADE_END_DATE = "2024-04-12"

df_raw = YahooDownloader(
    start_date=TRAIN_START_DATE,
    end_date=TRADE_END_DATE,
    ticker_list=config_tickers.DOW_30_TICKER,
).fetch_data()
print(df_raw.head())

fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_vix=True,
    use_turbulence=True,
    user_defined_feature=False,
)

processed = fe.preprocess_data(df_raw)

list_ticker = processed["tic"].unique().tolist()
list_date = list(
    pd.date_range(processed["date"].min(), processed["date"].max()).astype(str)
)
combination = list(itertools.product(list_date, list_ticker))

processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
    processed, on=["date", "tic"], how="left"
)
processed_full = processed_full[processed_full["date"].isin(processed["date"])]
processed_full = processed_full.sort_values(["date", "tic"])

processed_full = processed_full.fillna(0)
print(processed_full.head())

train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
print(len(train))
print(len(trade))

train.to_csv("data/train_data.csv")
trade.to_csv("data/trade_data.csv")
