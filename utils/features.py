import numpy as np
import pandas as pd

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['ret'] = out['Adj Close'].pct_change()
    out['logret'] = np.log(out['Adj Close']).diff()
    out['sma_5'] = out['Adj Close'].rolling(5).mean()
    out['sma_20'] = out['Adj Close'].rolling(20).mean()
    out['ema_12'] = out['Adj Close'].ewm(span=12, adjust=False).mean()
    out['ema_26'] = out['Adj Close'].ewm(span=26, adjust=False).mean()
    out['macd'] = out['ema_12'] - out['ema_26']
    delta = out['Adj Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-8)
    out['rsi_14'] = 100 - (100 / (1 + rs))
    out['vol_20'] = out['ret'].rolling(20).std()
    out['bb_mid'] = out['Adj Close'].rolling(20).mean()
    out['bb_std'] = out['Adj Close'].rolling(20).std()
    out['bb_width'] = (out['bb_std'] * 2) / (out['bb_mid'] + 1e-8)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out
