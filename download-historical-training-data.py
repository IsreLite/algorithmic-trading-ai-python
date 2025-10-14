import yfinance as yf

## Download BTC-USD historical data from Yahoo Finance
## Minute resolution data for the last 60 days
#data = yf.download(tickers='BTC-USD', period='1mo', interval='5m')
#data.to_csv('BTC-USD_historical_data.csv')

## auto-complete via GitHub Copilot

## Download News for BTC-USD from Yahoo Finance
## Last 60 days
news = yf.Ticker('BTC-USD').news
import pandas as pd
news_df = pd.DataFrame(news)
news_df.to_csv('BTC-USD_news.csv', index=False)
