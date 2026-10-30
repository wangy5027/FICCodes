# Tensor-based eLSTM for Stock Prediction

**Paper:** Qing Li, Jinghua Tan, Jun Wang, and Hsinchun Chen,  
“A Multimodal Event-Driven LSTM Model for Stock Prediction Using Online News,”  
*IEEE Transactions on Knowledge and Data Engineering*, 33(10), 2021, pp. 3323–3337.  
DOI: [10.1109/TKDE.2020.2968894](https://doi.org/10.1109/TKDE.2020.2968894)

## Overview

This work studies stock movement prediction by integrating financial fundamentals and online news information. The proposed framework uses tensor representations to capture interactions between information modalities and an event-driven LSTM architecture to account for the irregular arrival of news. A media-based enterprise network is also used to represent relationships among firms.

## Included files

- `models/model_RConvLSTM.py` — selected recurrent convolutional model definition from the original code repository.
- `models/model_ConvLSTM.py` — ConvLSTM model implementation included as a reference/baseline.
- `utils/RConvLSTM2D.py` — custom recurrent convolutional layer used by `model_RConvLSTM.py`.
- `data/stockdata_sample.csv` — a small sample of the stock-market data table from the original repository.

## Data fields

The sample contains stock code, closing/high/low/open prices, price-to-book ratio, price-to-earnings ratio, date, turnover, and trading volume. It is provided only as an illustrative data sample. It does not include the full financial-news data or all inputs used in the paper.
