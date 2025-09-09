import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from utils.data_fetch import fetch_prices, fetch_headlines_newsapi, compute_sentiment_embeddings
from utils.features import add_technical_indicators
from utils.modeling import build_sequences_multi, scale_sequences, train_sklearn_baseline, save_model, load_model_if_exists
from utils.backtest import backtest_signals

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

st.set_page_config(layout="wide", page_title="GenAI Trading (sklearn baseline)")
st.title("GenAI-Powered Trading — scikit-learn baseline (Demo)")
st.markdown("Educational demo. Not financial advice.")

# Sidebar
with st.sidebar:
    ticker = st.text_input("Ticker", value="AAPL")
    start_date = st.date_input("Start date", value=pd.to_datetime("2023-01-01"))
    end_date = st.date_input("End date", value=pd.to_datetime(pd.Timestamp.today().date()))
    seq_len = st.slider("lookback (days)", 10, 120, 30)
    horizon = st.slider("prediction horizon (days)", 1, 10, 5)
    model_type = st.selectbox("Model", ["RandomForest", "GradientBoosting"])
    retrain = st.button("Run / Train")
    use_newsapi = st.checkbox("Use NewsAPI headlines (optional)")
    newsapi_key = os.getenv('NEWSAPI_KEY', None)
    if use_newsapi and not newsapi_key:
        st.warning("NEWSAPI_KEY not found in environment; will use neutral embeddings.")

@st.cache_data
def cached_fetch_prices(ticker, start, end):
    return fetch_prices(ticker, start, end)

if not retrain:
    st.info("Set parameters and click Run / Train")
    st.stop()

# Pipeline
st.info("Fetching prices...")
prices = cached_fetch_prices(ticker, str(start_date), str(end_date))
st.success(f"Downloaded {len(prices)} rows")

st.info("Computing technical indicators...")
features = add_technical_indicators(prices)
st.dataframe(features.tail(3))

# Sentiment embeddings
if use_newsapi and newsapi_key:
    st.info("Fetching headlines...")
    headlines = fetch_headlines_newsapi(ticker, newsapi_key, str(start_date), str(end_date))
    emb_df = compute_sentiment_embeddings(features.index, headlines)
else:
    emb_dim = 8
    emb_df = pd.DataFrame(np.zeros((len(features), emb_dim)), index=features.index,
                          columns=[f"sent_{i}" for i in range(emb_dim)])
    st.info("Using neutral embeddings (zeros)")

feat_full = pd.concat([features, emb_df.reindex(features.index)], axis=1).dropna()
st.write("Feature matrix shape:", feat_full.shape)

st.info("Building sequences...")
X, y, y_idx = build_sequences_multi(feat_full, seq_len=seq_len, horizon=horizon)
if len(X) < 50:
    st.error("Not enough data. Extend date range or decrease seq_len.")
    st.stop()
st.success(f"{len(X)} sequences created")

# time split
split = int(0.85 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
idx_test = y_idx[split:]

X_train_s, X_test_s, scaler = scale_sequences(X_train, X_test)

# try loading existing model
model_path = MODEL_DIR / f"{ticker}_{model_type}.joblib"
model = load_model_if_exists(model_path)

if model is None:
    st.info("Training sklearn baseline...")
    model = train_sklearn_baseline(X_train_s, y_train, model_type=model_type)
    save_model(model, model_path)
    st.success(f"Model trained and saved to {model_path}")
else:
    st.info(f"Loaded existing model from {model_path}")

# predict
y_pred = model.predict(X_test_s)
rmse = np.sqrt(np.mean((y_test - y_pred)**2))
st.write(f"Test RMSE: {rmse:.6f}")

preds_series = pd.Series(index=y_idx, data=np.nan)
preds_series.loc[idx_test] = y_pred
preds_full = preds_series.reindex(feat_full.index).fillna(0.0)

st.info("Running backtest...")
bt_df, metrics = backtest_signals(feat_full.index, feat_full['Adj Close'], preds_full,
                                  threshold_long=0.01, threshold_short=-0.01,
                                  initial_capital=10000.0, position_size=0.1)
st.write(metrics)

st.subheader("Equity Curve")
st.line_chart(bt_df['nav'])

st.subheader("Latest prediction & recommendation")
latest_pred = preds_full.dropna().iloc[-1]
st.write(f"Latest predicted {horizon}-day return: {latest_pred:.2%}")
if latest_pred > 0.02:
    st.success("Bullish: consider small overweight with risk controls.")
elif latest_pred < -0.015:
    st.error("Bearish: consider reducing exposure or hedging.")
else:
    st.info("Neutral: observe and use risk controls.")
