import os
import joblib
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def get_macro_labels(model, scaler):
    means = scaler.inverse_transform(model.means_)
    labels = {}
    
    # Analyze all states to rank them relative to each other
    trends = [means[i][2]*100 for i in range(model.n_components)]
    vols = [means[i][1] for i in range(model.n_components)]
    
    for i in range(model.n_components):
        ret, vol, trend = means[i]
        ret_pct = (np.exp(ret) - 1) * 100
        trend_pct = trend * 100
        
        # Differentiate Extreme Bull (highest trend) from a regular Bull
        if trend_pct > 12:
            lbl = "Parabolic Bull / Euphoria"
        elif 5 < trend_pct <= 12:
            lbl = "Steady Bull Market"
        elif trend_pct < -5:
            lbl = "Bear Market"
        elif vol < 0.01:
            lbl = "Macro Accumulation"
        else:
            lbl = "Macro Chop"
            
        labels[i] = f"{lbl} (Ret:{ret_pct:.2f}%, dSMA:{trend_pct:.1f}%)"
    return labels

def get_micro_labels(model, scaler):
    means = scaler.inverse_transform(model.means_)
    labels = {}
    for i in range(model.n_components):
        ret, vol, vtrend = means[i]
        ret_pct = (np.exp(ret) - 1) * 100
        
        if vtrend > 1.5 and ret_pct > 0:
            lbl = "Exhaustion Pump"
        elif vtrend > 1.5 and ret_pct < 0:
            lbl = "Panic Dump"
        elif vol < 0.005:
            lbl = "Quiet Grind"
        else:
            # Split the remaining "Noise" states based on direction and volatility
            if ret_pct < 0:
                lbl = "Bearish Chop / Bleed"
            else:
                lbl = "Bullish Chop / Drift"
                
        labels[i] = f"{lbl} (Ret:{ret_pct:.2f}%, Vol:{vol:.3f})"
    return labels

def main():
    print("Loading Ensemble Models...")
    mac_mod = joblib.load('models/macro_hmm.pkl')
    mac_scl = joblib.load('models/macro_scaler.pkl')
    mic_mod = joblib.load('models/micro_hmm.pkl')
    mic_scl = joblib.load('models/micro_scaler.pkl')
    
    mac_labels = get_macro_labels(mac_mod, mac_scl)
    mic_labels = get_micro_labels(mic_mod, mic_scl)
    
    print("\n=== MACRO STATES (4H + 200 SMA Context) ===")
    for k, v in mac_labels.items(): print(f"Macro {k}: {v}")
        
    print("\n=== MICRO STATES (1H + Volume Profile Context) ===")
    for k, v in mic_labels.items(): print(f"Micro {k}: {v}")
        
    print("\n=== LIVE TRADING SIMULATION (Last 24 Hours) ===")
    df_mac = pd.read_csv('data/processed/BTCUSDT_macro.csv', index_col='open_time', parse_dates=True)
    df_mic = pd.read_csv('data/processed/BTCUSDT_micro.csv', index_col='open_time', parse_dates=True)
    
    # Predict histories
    df_mac['macro_state'] = mac_mod.predict(mac_scl.transform(df_mac[['log_return', 'volatility', 'trend_distance']].values))
    df_mic['micro_state'] = mic_mod.predict(mic_scl.transform(df_mic[['log_return', 'volatility', 'volume_trend']].values))
    
    # Merge Macro onto Micro (1H resolution with 4H overarching state)
    df_combo = df_mic[['close', 'micro_state']].copy()
    
    # Reindex macro to match micro's hourly index, using forward fill (ffill)
    # This means at 1:00, 2:00, 3:00, it uses the 0:00 Macro state.
    df_reindexed_mac = df_mac[['macro_state']].reindex(df_combo.index, method='ffill')
    df_combo['macro_state'] = df_reindexed_mac['macro_state']
    
    latest = df_combo.tail(24) # look at the last 24 hours
    
    print(f"{'Time':<20} | {'Price':<10} | {'MACRO ENVIRONMENT':<25} | {'MICRO TACTICAL ACTION'}")
    print("-" * 80)
    for idx, row in latest.iterrows():
        # Handle NAs at the very start if any exist
        if pd.isna(row['macro_state']):
            mac_str = "Unknown"
        else:
            mac_str = mac_labels.get(int(row['macro_state']), "Unknown").split('(')[0].strip()
            
        mic_str = mic_labels.get(int(row['micro_state']), "Unknown").split('(')[0].strip()
        
        print(f"{str(idx):<20} |  | {mac_str:<25} | {mic_str}")

if __name__ == "__main__":
    main()
