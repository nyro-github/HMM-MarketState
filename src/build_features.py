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
    
    # 1. Log Returns
    df_resampled['log_return'] = np.log(df_resampled['close'] / df_resampled['close'].shift(1))
    
    # 2. Realized Volatility 
    # Let's use a 24-period rolling window (e.g., 24 hours if resampled to 1h)
    volatility_window = 24
    df_resampled['volatility'] = df_resampled['log_return'].rolling(window=volatility_window).std()
    
    # 3. Volume Trend
    # Using the ratio of current volume to its rolling 24-period mean
    volume_window = 24
    df_resampled['volume_ma'] = df_resampled['volume'].rolling(window=volume_window).mean()
    # Adding a small epsilon to avoid division by zero
    epsilon = 1e-8
    df_resampled['volume_trend'] = df_resampled['volume'] / (df_resampled['volume_ma'] + epsilon)
    
    # Drop rows with NaNs originating from rolling windows and shifts
    df_resampled.dropna(inplace=True)
    
    # Optional: Keep only the features that will be used by the HMM, but we keep OHLCV for context/plotting
    
    return df_resampled

def main():
    db_path = os.path.join('data', 'raw', 'BTCUSDT.db')
    output_path = os.path.join('data', 'processed', 'BTCUSDT_features.csv')
    
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return
        
    df = load_data(db_path)
    df_features = process_features(df, resample_freq='1h') # Capital 'h' triggers a warning in new pandas, I'll use lowercase 'h'
    
    print(f"Saving processed features to {output_path}...")
    df_features.to_csv(output_path)
    print("Done! Here is a sample:")
    print(df_features[['log_return', 'volatility', 'volume_trend']].head())

if __name__ == "__main__":
    main()
