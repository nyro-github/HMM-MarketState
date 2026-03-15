import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random

def predict_states():
    data_path = os.path.join('data', 'processed', 'BTCUSDT_features.csv')
    df = pd.read_csv(data_path, index_col='open_time', parse_dates=True)
    
    scaler = joblib.load('models/feature_scaler.pkl')
    model = joblib.load('models/market_state_hmm.pkl')
    
    features = ['log_return', 'volatility', 'trend_distance']
    X = df[features].values
    X_scaled = scaler.transform(X)
    
    # Predict over the entire dataset first
    df['state'] = model.predict(X_scaled)
    return df, model

def plot_market_states(df_slice, iteration):
    print(f"Generating random chart {iteration}...")
    
    # Meaning assigned to each state based on our new Macro-trend features
    state_names = {
        0: "Quiet Bearish Drift",       # Negative return (-0.03%), low vol, slightly below SMA
        1: "Steady Uptrend",            # Positive return (+0.04%), low vol, moderately above SMA
        2: "High Volatility Markdown",  # Heavy negative return, high vol, deep (-7%) below SMA
        3: "Explosive Bull Market",     # Massive positive return, high vol, way above (+11%) SMA
    }
    
    # State 0 (Gray/Quiet Bear), State 1 (Green/Uptrend), State 2 (Red/Crash), State 3 (Blue/Explosive Bull)
    colors = ['gray', 'green', 'red', 'blue']
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    ax.plot(df_slice.index, df_slice['close'], color='black', alpha=0.6, linewidth=1, label='BTC Price')
    
    df_slice['state_change'] = df_slice['state'].diff().ne(0)
    state_blocks = df_slice[df_slice['state_change']].index
    
    for i in range(len(state_blocks) - 1):
        start = state_blocks[i]
        end = state_blocks[i+1]
        state = df_slice.loc[start, 'state']
        ax.axvspan(start, end, color=colors[state], alpha=0.3)
        
    last_start = state_blocks[-1]
    last_state = df_slice.loc[last_start, 'state']
    ax.axvspan(last_start, df_slice.index[-1], color=colors[last_state], alpha=0.3)
    
    start_date = df_slice.index[0].strftime('%Y-%m-%d')
    end_date = df_slice.index[-1].strftime('%Y-%m-%d')
    ax.set_title(f'BTC HMM 4-States: 30-Day Random Sample ({start_date} to {end_date})', fontsize=16)
    ax.set_ylabel('Price (USDT)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_yscale('log')
    
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=colors[i], alpha=0.3, label=f'{i}: {state_names[i]}') for i in range(4)]
    legend_patches.append(plt.Line2D([0], [0], color='black', lw=1, label='Price'))
    ax.legend(handles=legend_patches, loc='upper left')
    
    plt.tight_layout()
    
    output_dir = 'visualizations'
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'random_30d_sample_{iteration}.png')
    plt.savefig(filepath, dpi=300)
    plt.close(fig) # Prevent plot overlaps
    print(f"Chart saved to {filepath}")

def main():
    print("Loading data and inferring states over full history...")
    df, model = predict_states()
    
    window_size = 180 # 30 days * 6 (4H candles)
    max_idx = len(df) - window_size - 1
    
    # Generate 5 random samples
    for i in range(1, 6):
        start = random.randint(0, max_idx)
        df_slice = df.iloc[start:start+window_size].copy()
        plot_market_states(df_slice, i)

if __name__ == "__main__":
    main()
