import yfinance as yf
import requests
import numpy as np
import pandas as pd

def fetch_prices(ticker: str, start: str, end: str = None) -> pd.DataFrame:
    # Download data
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker} between {start} and {end}")

    # Handle MultiIndex (happens for multiple tickers or some yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level=1)  # extract single ticker level
        except KeyError:
            raise ValueError(f"Ticker '{ticker}' not found in downloaded data")

    # Keep only OHLCV (ignore missing ones for safety)
    needed = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    available = [c for c in needed if c in df.columns]
    df = df[available].copy()

    # Guarantee "Adj Close" exists
    if 'Adj Close' not in df.columns:
        if 'Close' in df.columns:
            df['Adj Close'] = df['Close']
        else:
            raise KeyError("Neither 'Adj Close' nor 'Close' found in data")

    # Drop missing rows
    df.dropna(inplace=True)

    return df



def fetch_headlines_newsapi(ticker: str, api_key: str, from_dt: str, to_dt: str, page_size:int=100):
    if not api_key:
        return {}
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': ticker,
        'from': from_dt,
        'to': to_dt,
        'language': 'en',
        'sortBy': 'relevancy',
        'pageSize': page_size,
        'apiKey': api_key
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return {}
        arts = r.json().get('articles', [])
        bydate = {}
        for a in arts:
            d = a.get('publishedAt','')[:10]
            t = a.get('title') or a.get('description') or ''
            if t:
                bydate.setdefault(d, []).append(t)
        return bydate
    except Exception:
        return {}

def compute_sentiment_embeddings(dates, headlines_by_date, model=None, emb_size=384):
    rows = []
    for dt in dates:
        key = dt.strftime('%Y-%m-%d')
        texts = headlines_by_date.get(key, [])
        if texts and model is not None:
            vecs = model.encode(texts, show_progress_bar=False)
            mean_vec = np.mean(vecs, axis=0)
        else:
            mean_vec = np.zeros(emb_size)
        rows.append(mean_vec)
    arr = np.vstack(rows)
    cols = [f'sent_{i}' for i in range(arr.shape[1])]
    return pd.DataFrame(arr, index=dates, columns=cols)
