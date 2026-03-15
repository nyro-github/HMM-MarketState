import sqlite3
import pandas as pd
import numpy as np
import os

def load_data(db_path, table_name='klines_1m'):
    """Load raw OHLCV data from SQLite database."""
    print(f"Loading data from {table_name}...")
    conn = sqlite3.connect(db_path)
    # Query opening time, OHLCV
    query = f"""
    SELECT open_time, open, high, low, close, volume 
    FROM {table_name}
    ORDER BY open_time ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convert timestamps
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    
    # Convert numeric columns
    cols = ['open', 'high', 'low', 'close', 'volume']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    
    return df

def process_features(df, resample_freq='1h'):
    """Resample and calculate log returns, volatility, and volume trends."""
    print(f"Resampling data to {resample_freq}...")
    
    # Define how to aggregate OHLCV data during resampling
    aggregation = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    df_resampled = df.resample(resample_freq).agg(aggregation)
    df_resampled.dropna(inplace=True)
    
    print("Calculating features...")
    
    # 1. Log Returns (Fast response direction)
    df_resampled['log_return'] = np.log(df_resampled['close'] / df_resampled['close'].shift(1))
    
    # 2. Realized Volatility 
    # Shortened to a rolling 48-hour window (12 periods if resampled to 4h)
    volatility_window = 12
    df_resampled['volatility'] = df_resampled['log_return'].rolling(window=volatility_window).std()
    
    # 3. Macro Trend Indicator (Distance from 100 SMA)
    # Using a 100-period Simple Moving Average on the 4H chart (roughly 16.6 days)
    sma_window = 100
    df_resampled['sma_100'] = df_resampled['close'].rolling(window=sma_window).mean()
    # Log distance from SMA (positive means Bull Trend, negative means Bear Trend)
    df_resampled['trend_distance'] = np.log(df_resampled['close'] / df_resampled['sma_100'])
    
    # Drop rows with NaNs originating from the long 200-period SMA requirement
    df_resampled.dropna(inplace=True)
    
    return df_resampled

def main():
    db_path = os.path.join('data', 'raw', 'BTCUSDT.db')
    output_path = os.path.join('data', 'processed', 'BTCUSDT_features.csv')
    
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return
        
    df = load_data(db_path)
    df_features = process_features(df, resample_freq='4h') 
    
    print(f"Saving processed features to {output_path}...")
    df_features.to_csv(output_path)
    print("Done! Here is a sample:")
    print(df_features[['log_return', 'volatility', 'trend_distance']].head())

if __name__ == "__main__":
    main()
