import numpy as np
import pandas as pd
import talib as ta
import random
import itertools
import yfinance as yf
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Preprocessor:
    def __init__(self):
        # PostgreSQL connection setup
        database_url = "postgresql+psycopg2://postgres@localhost:5432/sp500"
        self.engine = create_engine(database_url)

    def get_sp500(self) -> pd.DataFrame:
        # Read S&P500 Data
        query = "SELECT * FROM sp500_data_yahoo"
        data = pd.read_sql(query, self.engine)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        data = data.sort_index()
        return data

    def get_stock(self, name : str) -> pd.DataFrame:
        # Read stock Data
        query = f"SELECT date, high, low, close, volume FROM stock_data WHERE ticker = '{name}'"
        data = pd.read_sql(query, self.engine)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        data = data.sort_index()
        return data

    @staticmethod
    def match_volatility(df_train: pd.DataFrame, df_test : pd.DataFrame) -> tuple:
        volatility = df_train.std()
        target_volatility = volatility.mean()
        df_train = df_train.apply(lambda x: x * (target_volatility / x.std()), axis=0)
        # Use the training set volatility to adjust the test set
        df_test = df_test.apply(lambda x: x * (target_volatility / volatility[x.name]), axis=0)
        return df_train, df_test

    @staticmethod
    def generate_technicals(df: pd.DataFrame) -> pd.DataFrame:
        # Extract the necessary columns
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']

        # Calculate various technical indicators
        rsi = ta.RSI(close, timeperiod=30)
        atr = ta.ATR(high, low, close, timeperiod=30)
        mom = ta.MOM(close, timeperiod=30)
        obv = ta.OBV(close, volume)
        slowk, slowd = ta.STOCH(high, low, close, fastk_period=45, slowk_period=30, slowk_matype=0, slowd_period=3, slowd_matype=0)
        macd, macd_signal, macd_hist = ta.MACD(close, fastperiod=45, slowperiod=30, signalperiod=9)
        ad = ta.AD(high, low, close, volume)
        upper_band, middle_band, lower_band = ta.BBANDS(close, timeperiod=30, nbdevup=2, nbdevdn=2, matype=0)
        width_bollinger_band = (upper_band - lower_band) / middle_band * 100

        # Create a DataFrame with the technical indicators
        df_technicals = pd.DataFrame({
            'ATR': atr,
            'RSI': rsi,
            'MOM': mom,
            'OBV': obv,
            'STOCH_K': slowk,
            'STOCH_D': slowd,
            'MACD': macd,
            'MACD_Signal': macd_signal,
            'MACD_Hist': macd_hist,
            'AD': ad,
            'Bollinger_Band_width': width_bollinger_band
        })

        df_technicals = df_technicals.dropna()

        return df_technicals

    @staticmethod
    def get_combinations(df_technicals: pd.DataFrame, n_features: int = 2) -> list:
        indicators = df_technicals.columns.tolist()
        random.shuffle(indicators)
        used_indicators = set()
        valid_combinations = []

        # Generate all pairs of technical indicators
        for combo in itertools.combinations(indicators, n_features):
            if combo[0] not in used_indicators and combo[0] not in used_indicators:
                # If neither indicator has been used, add the combo and mark them as used
                valid_combinations.append(combo)
                used_indicators.update(combo)
        return valid_combinations

    def get_stock_by_name(self, stock_name: str, start='2016-01-04', end='2024-09-07') -> pd.DataFrame:
        """Download stock data if not already in the database."""
        if self.__stock_exists_in_db(stock_name):
            print(f"{stock_name} already exists in the database. Skipping download.")
            return pd.DataFrame()  # Return an empty DataFrame to signify no new data
        else:
            # Download the stock data from Yahoo Finance
            print(f"Downloading data for {stock_name}.")
            data = yf.download(stock_name, start=start, end=end)
            return data

    def __stock_exists_in_db(self, stock_name: str) -> bool:
        """Check if the stock is already in the database."""
        query = f"SELECT 1 FROM stock_data WHERE ticker = '{stock_name}' LIMIT 1"
        result = pd.read_sql(query, self.engine)
        return not result.empty

    def upload_to_postgres(self, data: pd.DataFrame, stock_name: str):
        """Upload the stock data to the PostgreSQL database."""
        if not data.empty:
            data['ticker'] = stock_name  # Add stock ticker to the DataFrame
            data.to_sql('stock_data', self.engine, if_exists='append', index=True, index_label='date')
            print(f"Uploaded {stock_name} data to the database.")