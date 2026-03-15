import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings

warnings.filterwarnings('ignore')

# Import the dynamic label generators we built in the analyzer
from ensemble_analyzer import get_macro_labels, get_micro_labels

def load_data_and_predict():
    # Load Models & Scalers
    mac_mod = joblib.load('models/macro_hmm.pkl')
    mac_scl = joblib.load('models/macro_scaler.pkl')
    mic_mod = joblib.load('models/micro_hmm.pkl')
    mic_scl = joblib.load('models/micro_scaler.pkl')
    
    # Load Data
    df_mac = pd.read_csv('data/processed/BTCUSDT_macro.csv', index_col='open_time', parse_dates=True)
    df_mic = pd.read_csv('data/processed/BTCUSDT_micro.csv', index_col='open_time', parse_dates=True)
    
    # Predict Full History
    df_mac['state'] = mac_mod.predict(mac_scl.transform(df_mac[['log_return', 'volatility', 'trend_distance']].values))
    df_mic['state'] = mic_mod.predict(mic_scl.transform(df_mic[['log_return', 'volatility', 'volume_trend']].values))
    
    return df_mac, df_mic, mac_mod, mac_scl, mic_mod, mic_scl

def plot_hierarchical(df_mac, df_mic, mac_labels, mic_labels, start_date, end_date, iteration):
    print(f"Plotting Dual-Pane Chart {iteration} from {start_date.date()} to {end_date.date()}...")
    
    # Slice the data
    mac_slice = df_mac.loc[start_date:end_date].copy()
    mic_slice = df_mic.loc[start_date:end_date].copy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    
    # Standard colors to assign to the 4 states arbitrarily
    # The legend will map the color to the dynamically generated text from the analyzer
    c_palette = {0: '#808080', 1: '#2CA02C', 2: '#FF4500', 3: '#1F77B4'} # Gray, Green, Orange, Blue
    
    # --- TOP PANE: 4H MACRO ---
    ax1.plot(mac_slice.index, mac_slice['close'], color='black', alpha=0.8, linewidth=1.5, label='BTC Price (4H)')
    mac_slice['state_change'] = mac_slice['state'].diff().ne(0)
    blocks = mac_slice[mac_slice['state_change']].index
    
    for i in range(len(blocks) - 1):
        s_idx, e_idx = blocks[i], blocks[i+1]
        st = mac_slice.loc[s_idx, 'state']
        ax1.axvspan(s_idx, e_idx, color=c_palette[st], alpha=0.3)
    ax1.axvspan(blocks[-1], mac_slice.index[-1], color=c_palette[mac_slice.loc[blocks[-1], 'state']], alpha=0.3)
    
    ax1.set_title(f"MACRO Regime (4H DataFrame) - Overarching Trend", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Price (USDT)", fontsize=12)
    ax1.set_yscale('log')
    
    import matplotlib.patches as mpatches
    mac_patches = [mpatches.Patch(color=c_palette[i], alpha=0.3, label=f"State {i}: {mac_labels[i].split('(')[0]}") for i in range(4)]
    ax1.legend(handles=mac_patches, loc='upper left')

    # --- BOTTOM PANE: 1H MICRO ---
    ax2.plot(mic_slice.index, mic_slice['close'], color='black', alpha=0.6, linewidth=1, label='BTC Price (1H)')
    mic_slice['state_change'] = mic_slice['state'].diff().ne(0)
    blocks = mic_slice[mic_slice['state_change']].index
    
    for i in range(len(blocks) - 1):
        s_idx, e_idx = blocks[i], blocks[i+1]
        st = mic_slice.loc[s_idx, 'state']
        ax2.axvspan(s_idx, e_idx, color=c_palette[st], alpha=0.3)
    ax2.axvspan(blocks[-1], mic_slice.index[-1], color=c_palette[mic_slice.loc[blocks[-1], 'state']], alpha=0.3)
    
    ax2.set_title(f"MICRO Tactical (1H DataFrame) - Fast Execution & Noise", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Price (USDT)", fontsize=12)
    ax2.set_yscale('log')
    
    mic_patches = [mpatches.Patch(color=c_palette[i], alpha=0.3, label=f"State {i}: {mic_labels[i].split('(')[0]}") for i in range(4)]
    ax2.legend(handles=mic_patches, loc='upper left')

    plt.tight_layout()
    
    out_dir = 'visualizations'
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, f'dual_tier_sample_{iteration}.png')
    plt.savefig(filepath, dpi=300)
    plt.close(fig)
    print(f"-> Saved {filepath}")

def main():
    print("Loading data...")
    df_mac, df_mic, mac_mod, mac_scl, mic_mod, mic_scl = load_data_and_predict()
    
    mac_labels = get_macro_labels(mac_mod, mac_scl)
    mic_labels = get_micro_labels(mic_mod, mic_scl)
    
    # Generate 3 random 45-day slices for visualizing the dual relationship
    window_days = 45
    min_date = df_mac.index.min() + pd.Timedelta(days=1)
    max_date = df_mac.index.max() - pd.Timedelta(days=window_days)
    
    # Pre-select some interesting historical periods if random isn't preferred,
    # but we will do 3 fully random ones.
    for i in range(1, 4):
        random_start = min_date + (max_date - min_date) * random.random()
        random_end = random_start + pd.Timedelta(days=window_days)
        plot_hierarchical(df_mac, df_mic, mac_labels, mic_labels, random_start, random_end, i)

if __name__ == "__main__":
    main()
