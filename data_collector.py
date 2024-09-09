import pandas as pd
import yfinance as yf

class DataCollector():
    def __init__(self) -> None:
        self.tickers = None
        self.raw_data = None

    def set_tickers(self, new_tickers):
        self.tickers = new_tickers

    def download_data(self, start, end, interval):
        self.raw_data = yf.download(self.tickers, start, end, interval)

    def get_raw_data(self):
        return self.raw_data