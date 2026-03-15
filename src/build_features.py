import sqlite3
import pandas as pd
import numpy as np
import os

def load_data(db_path, table_name='klines_1m'):
    print(f"Loading data from {table_name}...")
    conn = sqlite3.connect(db_path)
    query = f"SELECT open_time, open, high, low, close, volume FROM {table_name} ORDER BY open_time ASC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    cols = ['open', 'high', 'low', 'close', 'volume']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    return df

def process_macro(df):
    print("Building Macro Features (4H)...")
    agg = {'open': 'first', 'high': 'max', 'low': 'min','close': 'last', 'volume': 'sum'}
    df_4h = df.resample('4h').agg(agg).dropna()
    
    df_4h['log_return'] = np.log(df_4h['close'] / df_4h['close'].shift(1))
    df_4h['volatility'] = df_4h['log_return'].rolling(window=42).std()
    df_4h['sma_200'] = df_4h['close'].rolling(window=200).mean()
    df_4h['trend_distance'] = np.log(df_4h['close'] / df_4h['sma_200'])
    
    return df_4h.dropna()

def process_micro(df):
    print("Building Micro Features (1H)...")
    agg = {'open': 'first', 'high': 'max', 'low': 'min','close': 'last', 'volume': 'sum'}
    df_1h = df.resample('1h').agg(agg).dropna()
    
    df_1h['log_return'] = np.log(df_1h['close'] / df_1h['close'].shift(1))
    df_1h['volatility'] = df_1h['log_return'].rolling(window=24).std()
    
    df_1h['volume_ma'] = df_1h['volume'].rolling(window=24).mean()
    df_1h['volume_trend'] = df_1h['volume'] / (df_1h['volume_ma'] + 1e-8)
    
    return df_1h.dropna()

def main():
    db_path = os.path.join('data', 'raw', 'BTCUSDT.db')
    out_macro = os.path.join('data', 'processed', 'BTCUSDT_macro.csv')
    out_micro = os.path.join('data', 'processed', 'BTCUSDT_micro.csv')
    
    df = load_data(db_path)
    
    df_macro = process_macro(df)
    df_macro.to_csv(out_macro)
    print(f"Macro 4H dataset saved. Shape: {df_macro.shape}")
    
    df_micro = process_micro(df)
    df_micro.to_csv(out_micro)
    print(f"Micro 1H dataset saved. Shape: {df_micro.shape}")

if __name__ == "__main__":
    main()
