import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import joblib
import warnings

warnings.filterwarnings("ignore")

def train_pipeline(data_path, features, output_prefix, n_components=4, n_iter=1000):
    print(f"\n--- Training {output_prefix.upper()} Model ---")
    df = pd.read_csv(data_path, index_col='open_time', parse_dates=True)
    X = df[features].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=n_iter, random_state=42, tol=0.01)
    model.fit(X_scaled)
    print(f"Converged: {model.monitor_.converged} (Log-Likelihood: {model.monitor_.history[-1]:.2f})")
    
    joblib.dump(model, f'models/{output_prefix}_hmm.pkl')
    joblib.dump(scaler, f'models/{output_prefix}_scaler.pkl')
    
    return model, scaler

def main():
    os.makedirs('models', exist_ok=True)
    
    # Train Macro (4H) Model
    macro_feats = ['log_return', 'volatility', 'trend_distance']
    train_pipeline('data/processed/BTCUSDT_macro.csv', macro_feats, 'macro', n_components=4)
    
    # Train Micro (1H) Tactical Model
    micro_feats = ['log_return', 'volatility', 'volume_trend']
    train_pipeline('data/processed/BTCUSDT_micro.csv', micro_feats, 'micro', n_components=4)

if __name__ == "__main__":
    main()
