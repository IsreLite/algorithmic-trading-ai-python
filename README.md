# Algorithmic Trading AI with Python

## Tuning and Optimization
 - Calcluate the Cost
 - Reduce dimentionality of embeddings
 - Change input data add more Symbols
 - EARLY STOPPING!!!
 - ✅ More data again!
 - Labels threshold changes ( reduce from 0.1% to 0.05% )
 - Batch Size
     - Multi-stage Batch Training
 - Dropout Rate
 - Learning Rate
 - ✅ SGD
 - ✅ Activations
 - More Input
    - Time of day (vectorize, day of week )
 - Learning rate adjustments
 - ✅ More DATA!
     - ✅ More Data ( pull in from download.py)
     - ✅
 - Sequence Data ( merge sequences together )
 - Embedding Cacheing ( save embeddings hash )
 - ~~Instead of Gemma UniformScalar + NLP Embedding~~

## Target Market Pattern
 - Volatile Market
 - Lots of news events 
 - High volume

![Volatile Market](target-profit.png)

## Libraries
 - Yfinance (datasource)
 - Pytorch Transformer 
 - Gemma embedding 300m GOOGLE 
 - Numpy
 - matplotlib
 - Pandas
 - Scikit-learn

## Target Symbol
 - BTC-USD (Bitcoin to US Dollar)
 - AAPL (Apple Inc.)
 - TSLA (Tesla Inc.)
 - GC=F (Gold Futures)

## Inputs and outputs

### Input data
 - Historical price data 
 - News headlines sentiment will be derived
 - Foreign policy events

### Outputs
 - Sell/Hold/Buy signals

## Model
 - Gemma Embedding (300m parameters)
 - Pytorch Transformer
 - Classification Layer ([3] output)

## Training
 - Supervised learning
 - Reinforcement learning

### Input Data Format 

Sorted by date, descending.

```
Prices: list of prices
Headline: list of news headlines
```

```
Prices: 10.0,11.0,14.0,8.0,10.0,5.0
Headline: Sector Update: Financial Stocks Rise Tuesday Afternoon

Prices: 10.0,11.0,14.0,8.0,10.0,5.0
Headline: Sector Update: Financial Stocks Rise Tuesday Afternoon

Prices: 10.0,11.0,14.0,8.0,10.0,5.0
Headline: Sector Update: Financial Stocks Rise Tuesday Afternoon
```
### Ouptput Data Format
Sell/Hold/Buy signals
[0,   0,   0 ]

Sell = [1, 0, 0]
Hold = [0, 1, 0]
Buy  = [0, 0, 1]

### Limit Triggers Indications
- Average up and down cycles
- 1% limit triggers 100 gain 1 dollar 101
- time window is 5 minutes
