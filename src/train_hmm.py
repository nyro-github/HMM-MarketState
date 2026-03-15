import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import joblib
import warnings

warnings.filterwarnings("ignore")

def load_and_prepare_data(filepath):
    """
    Phase 1 Output -> Phase 2 Input Strategy
    """
    print(f"Loading feature data from {filepath}...")
    df = pd.read_csv(filepath, index_col='open_time', parse_dates=True)
    
    # Selected Features out of the DataFrame (Phase 2)
    features = ['log_return', 'volatility', 'trend_distance']
    X = df[features].values
    
    # Phase 3: Data Scaling
    # Essential for GaussianHMM to converge when features are of different scales
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, df, scaler, features

def train_hmm(X, n_components=4, n_iter=1000, random_state=42):
    """
    Phase 2: Model Architecture
    Phase 3: Model Fitting
    """
    print(f"Initializing GaussianHMM with n_components={n_components}...")
    
    # We use full covariance matrices since return and volatility likely have correlation
    model = GaussianHMM(
        n_components=n_components, 
        covariance_type="full", 
        n_iter=n_iter, 
        random_state=random_state,
        tol=0.01  # Convergence tolerance
    )
    
    print("Training the HMM model (this may take a moment)...")
    # EM algorithm to fit model
    model.fit(X)
    
    if model.monitor_.converged:
        print(f"Model successfully converged! (Log-Likelihood: {model.monitor_.history[-1]:.2f})")
    else:
        print("Warning: Model did not perfectly converge. Consider increasing n_iter or tuning tolerance.")
        
    return model

def main():
    data_path = os.path.join('data', 'processed', 'BTCUSDT_features.csv')
    model_dir = 'models'
    
    # Create models directory
    os.makedirs(model_dir, exist_ok=True)

    # 1. Load Data & Scale
    X_scaled, df, scaler, feature_names = load_and_prepare_data(data_path)
    
    # 2. Train Model with 4 states
    model = train_hmm(X_scaled, n_components=4)
    
    # 3. Save Model & Scaler for Interpretations/Predictions later
    model_path = os.path.join(model_dir, 'market_state_hmm.pkl')
    scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Saved optimized model   -> {model_path}")
    print(f"Saved feature scaler    -> {scaler_path}")

if __name__ == "__main__":
    main()
