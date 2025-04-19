import json
import pandas as pd
import yfinance as yf
import ta
from datetime import datetime


def backtest_strategy(df, signal, start, end, init_cap=100.0):
    """
    signal: pandas Series of 0/1 indicating position at close of each day
    entry on 1 -> hold long, exit on 0 -> cash
    calculate next day returns when in position
    returns: pandas Series of cumulative equity starting from init_cap
    """
    p = df.loc[start:end].copy()
    p['signal']    = signal.loc[start:end]
    p['next_ret']  = p['Close'].pct_change().shift(-1).fillna(0)
    p['strat_ret'] = p['signal'] * p['next_ret']
    p['equity']    = init_cap * (1 + p['strat_ret']).cumprod()
    return p['equity']


def investor1_signal(df):
    # 10-day momentum: hold long whenever 10-day returns positive
    close = df['Close'].squeeze()
    mom = close.pct_change(10)
    return (mom > 0).astype(int)


def investor2_signal(df):
    # RSI strategy: enter when RSI < 30, exit when RSI > 70
    close = df['Close'].squeeze()
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
    signal = pd.Series(0, index=df.index)
    position = 0
    for i in range(len(df)):
        if position == 0 and rsi.iat[i] < 30:
            position = 1
        elif position == 1 and rsi.iat[i] > 70:
            position = 0
        signal.iat[i] = position
    return signal


def investor3_signal(df):
    # SMA 5/20 crossover: golden cross -> enter, death cross -> exit
    close = df['Close'].squeeze()
    sma5 = close.rolling(window=5).mean()
    sma20 = close.rolling(window=20).mean()
    signal = pd.Series(0, index=df.index)
    position = 0
    for i in range(1, len(df)):
        # golden cross
        if position == 0 and sma5.iat[i-1] <= sma20.iat[i-1] and sma5.iat[i] > sma20.iat[i]:
            position = 1
        # death cross
        elif position == 1 and sma5.iat[i-1] >= sma20.iat[i-1] and sma5.iat[i] < sma20.iat[i]:
            position = 0
        signal.iat[i] = position
    return signal


def main():
    tickers = ['AAPL', 'NVDA', 'TSLA', 'GOOGL']
    start = '2025-01-01'
    end   = '2025-04-15'

    for t in tickers:
        # download data
        df = yf.download(t, start='2020-01-01', end=end)[['Close']].dropna()

        # investor 1
        sig1 = investor1_signal(df)
        eq1  = backtest_strategy(df, sig1, start, end, init_cap=100.0)
        # investor 2
        sig2 = investor2_signal(df)
        eq2  = backtest_strategy(df, sig2, start, end, init_cap=100.0)
        # investor 3
        sig3 = investor3_signal(df)
        eq3  = backtest_strategy(df, sig3, start, end, init_cap=100.0)

                # assemble into date->values dict with returns relative to initial cap (100)
        results = {}
        for date, v1, v2, v3 in zip(eq1.index, eq1.values, eq2.values, eq3.values):
            ds = date.strftime('%Y-%m-%d')
            results[ds] = {
                'investor1_mom': v1 - 100.0,
                'investor2_rsi': v2 - 100.0,
                'investor3_sma': v3 - 100.0
            }

        # write separate JSON per ticker
        fname = f"{t}_investor_results.json"
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Saved results for {t} to {fname}")

if __name__ == '__main__':
    main()
