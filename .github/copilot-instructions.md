# AI Agent Instructions for algorithmic-trading-ai-python

## Project Overview
This is a PyTorch-based algorithmic trading system that combines Google's Gemma embeddings with a transformer classifier to generate trading signals (Buy/Hold/Sell) based on price data and news headlines.

## Key Components and Architecture

### Core Model Structure (`models/gemma_transformer_classifier.py`)
- `SimpleGemmaTransformerClassifier`: Main model class combining Gemma embeddings and transformer
- Uses Google's Gemma 300m model for text embeddings with embedding caching
- Custom transformer encoder for classification (3 classes: Buy/Hold/Sell)
- Automatically selects best available device (CUDA GPU → Apple Silicon MPS → CPU)

### Data Flow
1. Input format combines price data and news:
```
Price: <price_values>
Headline: <news_headline>
Summary: <news_summary>
```
2. Text gets embedded via Gemma model (cached using SHA256 hash)
3. Embeddings processed through transformer encoder
4. Output is 3-way classification: [Sell, Hold, Buy] as [1,0,0], [0,1,0], [0,0,1]

## Development Workflows

### Environment Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Authenticate with HuggingFace: `hf auth login` (required for Gemma model)

### Training Pipeline
- Data split: 80% training, 20% testing
- Default hyperparameters (in `train.py`):
  - Learning rate: 0.005
  - Batch size: 1
  - Epochs: 20
- Model saves to `gemma_transformer_classifier.pth`

### Signal Generation
Signal thresholds:
- Buy: price increase > 1%
- Sell: price decrease < -1%
- Hold: price change between -1% and 1%

## Project Conventions

### Model Training Pattern
1. Data preparation combines price and news data from JSON files
2. Uses SGD optimizer with CrossEntropyLoss
3. Implements stochastic batch selection
4. Tracks rolling average loss over last 250 items

### Caching Strategy
- Embeddings are cached using SHA256 hash of input text
- Cache persists during model lifetime to improve performance

## Integration Points
- Primary data source: yfinance (for price and websocket data)
- Text embedding: google/embeddinggemma-300m via SentenceTransformer
- Data files: Expects JSON files for historical data and news
  - `BTC-USD_historical_data.json`
  - `BTC-USD_news_with_price.json`
  - `BTC-USD_news.json`

## Current Status and Limitations
- Optimized for volatile markets with high news volume
- Target symbols: BTC-USD, AAPL, TSLA, GC=F
- 5-minute trading window
- Model assumes significant correlation between news sentiment and price movement