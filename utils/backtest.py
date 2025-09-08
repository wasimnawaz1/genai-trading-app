import numpy as np
import pandas as pd

def backtest_signals(dates, close_prices, preds,
                     threshold_long=0.01, threshold_short=-0.01,
                     initial_capital=10000.0, position_size=0.1,
                     tc=0.0005, slippage=0.0005):
    df = pd.DataFrame(index=dates)
    df['close'] = close_prices.reindex(dates).ffill()
    df['pred'] = preds.reindex(dates).fillna(0)
    df['signal'] = 0
    df.loc[df['pred'] >= threshold_long, 'signal'] = 1
    df.loc[df['pred'] <= threshold_short, 'signal'] = -1

    cash = initial_capital
    pos = 0.0
    nav_list = []
    pos_list = []
    trades = 0

    for i in range(len(df)-1):
        price = df['close'].iloc[i]
        sig = df['signal'].iloc[i]
        target_value = initial_capital * position_size * sig
        current_value = pos * price
        delta = target_value - current_value
        shares = delta / price if price>0 else 0.0
        if abs(shares) > 1e-9:
            trade_cost = abs(shares) * price * (tc + slippage)
            cash -= shares * price + trade_cost
            pos += shares
            trades += 1
        nav = cash + pos * price
        nav_list.append(nav)
        pos_list.append(pos)
    nav_list.append(cash + pos * df['close'].iloc[-1])
    pos_list.append(pos)

    out = pd.DataFrame(index=df.index)
    out['nav'] = nav_list
    out['pos'] = pos_list
    out['close'] = df['close']
    out['pred'] = df['pred']
    out['signal'] = df['signal']
    out['returns'] = out['nav'].pct_change().fillna(0)

    total_return = out['nav'].iloc[-1] / out['nav'].iloc[0] - 1
    ann_ret = (1 + total_return) ** (252.0 / len(out)) - 1
    ann_vol = out['returns'].std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-8)
    cum = out['nav']
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min()

    metrics = {
        'total_return': float(total_return),
        'annualized_return': float(ann_ret),
        'annualized_vol': float(ann_vol),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'trades': int(trades)
    }
    return out, metrics
