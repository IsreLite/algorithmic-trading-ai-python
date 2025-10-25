# AI Agent Guide: algorithmic-trading-ai-python

## Core Architecture
- Single model pipeline: Text+Price → Embeddings → Transformer → Trading Signal
- Model implementation in `models/gemma_transformer_classifier.py`
- Automatic hardware selection: CUDA → MPS → CPU

## Key Data Patterns
1. Input format (see `train.py`):
```python
features = [
    f"Price: {price}\n" +
    f"Headline: {headline}\n" +
    f"Summary: {summary}"
]
```

2. Signal encoding (see `train.py`):
```python
sell = [1., 0., 0.]  # price_change < -0.01
hold = [0., 1., 0.]  # -0.01 ≤ price_change ≤ 0.01
buy  = [0., 0., 1.]  # price_change > 0.01
```

## Critical Workflows

### First-time Setup
```bash
pip install -r requirements.txt
hf auth login  # Required for Gemma model access
```

### Data Pipeline (`download.py` → `train.py` → `test.py`)
1. `download.py`: Fetches fresh market data (optional)
2. `train.py`: 80/20 train/test split, saves to `gemma_transformer_classifier.pth`
3. `test.py`: Evaluation and signal generation

### Live Trading Implementation
- Copy `test.py` → `profit.py`
- Modify to use yfinance websocket stream
- Follow signal processing pattern from `test.py`

## Performance Optimizations
- Embedding cache using SHA256 text hashing
- Rolling loss window (250 items) for training stability
- Stochastic batch selection

## Key Dependencies
- `sentence_transformers`: Gemma 300m embedding model
- `torch`: Model architecture and training
- `yfinance`: Market data source

## Current Focus
- 5-minute trading window
- BTC-USD primary target (also supports AAPL, TSLA, GC=F)
- Optimized for high-volume, news-driven markets