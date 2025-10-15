import json
import datetime
import yfinance as yf

def main():
    prepare_data()
    #download_ticker()

## Download BTC-USD historical data from Yahoo Finance
## Minute resolution data for the last 60 days
def download_ticker():
    data = yf.download(tickers='BTC-USD', period='1mo', interval='5m')
    encoded = data.to_json()
    decoded = json.loads(encoded)
    close   = decoded["('Open', 'BTC-USD')"]
    
    ## Save to file
    with open('BTC-USD_historical_data.json', 'w') as f:
        json.dump(close, f, indent=4)
        

def download_news():
    ## Download News for BTC-USD from Yahoo Finance
    news = yf.Ticker('BTC-USD').get_news(count=1000)
    with open('BTC-USD_news.json', 'w') as f:
        json.dump(news, f, indent=4)

## Prepare data for training
def prepare_data():
    output = []
    with open('BTC-USD_historical_data.json', 'r') as f:
        ticker = json.load(f)
        #print(json.dumps(ticker, indent=4)[:5])  # Print first 500 characters

    with open('BTC-USD_news.json', 'r') as f:
        news = json.load(f)
        #print(json.dumps(news, indent=4)[:500])  # Print first 500 characters

    ## Augment with Pricing data
    for item in news:
        title   = item['content']['title']
        summary = item['content']['summary']
        pubDate = item['content']['pubDate']
        ## Convert pubDate to unix timestamp
        pubDate_ts = int(datetime.datetime.strptime(pubDate, '%Y-%m-%dT%H:%M:%SZ').timestamp())

        # Round down to nearest 5 minutes
        index = pubDate_ts - (pubDate_ts % 300)  
        price = ticker.get(f"{index}000")

        output.append({
            'title': title,
            'index': index,
            'price' : price,
            'summary': summary,
            'pubDate': pubDate,
            'pubDate_ts': pubDate_ts,
        })
        print(output)
        return

    with open('BTC-USD_news_with_price.json', 'w') as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__": main()
