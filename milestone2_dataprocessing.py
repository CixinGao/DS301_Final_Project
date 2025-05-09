import yfinance as yf
from fredapi import Fred
import pandas as pd
import numpy as np

# Gold (GLD ETF or use 'GC=F' for futures)
t='2001-01-01'
gold = yf.download('GC=F', start=t, end='2024-01-01')

# USD Index
dxy = yf.download('DX-Y.NYB', start=t, end='2024-01-01')
sp500 = yf.download('^GSPC', start=t)


fred = Fred(api_key='77a9fc2337f451148a39860436ecfe13')

# CPI (Consumer Price Index)
cpi = fred.get_series('CPIAUCSL', realtime_start=t, realtime_end='2024-01-01')

# Federal Funds Rate
fed_rate = fred.get_series('FEDFUNDS', realtime_start=t, realtime_end='2024-01-01')

series_list = {
    "Gold Price": gold['Close'],
    "USD Index": dxy['Close'],
    "CPI": cpi,
    "Fed Rate": fed_rate
}


import pandas as pd

# Step 1: Prepare all series with datetime index and filter from 2010
gold_series = pd.Series(gold['Close'].values.squeeze(), index=pd.to_datetime(gold.index)).loc[t:].rename("Gold Price")
sp500_series = pd.Series(sp500['Close'].values.squeeze(), index=pd.to_datetime(sp500.index)).loc[t:].rename("sp500")
usd_series = pd.Series(dxy['Close'].values.squeeze(), index=pd.to_datetime(dxy.index)).loc[t:].rename("USD Index")
cpi_series = pd.Series(cpi, name="CPI").loc[t:]
fed_series = pd.Series(fed_rate, name="Fed Rate").loc[t:]

# Step 2: Use gold price index (market open days) as the master timeline
date_range = gold_series.dropna().index

# Step 3: Reindex everything to gold’s trading days, using forward-fill
usd_series = usd_series.reindex(date_range).ffill()
cpi_series = cpi_series.reindex(date_range).ffill()
sp500_series = sp500_series.reindex(date_range).ffill()
fed_series = fed_series.reindex(date_range).ffill()

# Step 4: Combine all series into one DataFrame
combined_df = pd.concat([gold_series, usd_series, cpi_series, fed_series,sp500_series], axis=1)

# Step 5: Drop any remaining NaNs just in case (shouldn’t happen, but safe)
combined_df = combined_df.dropna()

# Step 6: Save to Excel
combined_df.to_excel('gold_macro_trading_days.xlsx', index_label='Date')

print("✅ Final dataset with market open days only saved to 'gold_macro_trading_days.xlsx'")
