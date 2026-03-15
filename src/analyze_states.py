import joblib
import pandas as pd
import numpy as np

def main():
    print("Loading model and scaler...")
    model = joblib.load('models/market_state_hmm.pkl')
    scaler = joblib.load('models/feature_scaler.pkl')
    
    # Model means are in the scaled feature space. Let's convert them back to original scale.
    original_means = scaler.inverse_transform(model.means_)
    
    features = ['log_return', 'volatility', 'trend_distance']
    df_means = pd.DataFrame(original_means, columns=features)
    
    # Let's also grab the variances for each state (the diagonal of the covariance matrix)
    # The covars_ array shape is (n_components, n_features, n_features) if covariance_type="full"
    variances = []
    for i in range(model.n_components):
        # We just want the diagonal (variance of each feature)
        # Note: These variances are in the *scaled* space. We'll just look at the raw means first.
        diag = np.diag(model.covars_[i])
        variances.append(diag)
        
    df_means.index.name = 'State'
    
    print("\n" + "="*50)
    print("Market State Characteristics (Original Feature Scale)")
    print("="*50)
    # Convert log return to percentage for better readability
    df_means['return_pct_per_hr'] = (np.exp(df_means['log_return']) - 1) * 100
    print(df_means[['return_pct_per_hr', 'volatility', 'trend_distance']])
    
    print("\n" + "="*50)
    print("State Persistence (Probability of staying in the same state next hour)")
    print("="*50)
    persistence = np.diag(model.transmat_)
    for i, p in enumerate(persistence):
        print(f"State {i}: {p*100:.2f}%")
        
    print("\n" + "="*50)
    print("State Frequencies (Stationary Distribution)")
    print("="*50)
    try:
        # Stationary distribution can sometimes be fetched like this or via an eigenvector
        eigvals, eigvecs = np.linalg.eig(model.transmat_.T)
        stationary = np.array(eigvecs[:, np.where(np.abs(eigvals - 1.) < 1e-8)[0][0]].flat)
        stationary = stationary / np.sum(stationary)
        for i, p in enumerate(stationary.real):
            print(f"State {i}: {p*100:.2f}% of the time")
    except Exception as e:
        print("Could not calculate stationary distribution directly.")

if __name__ == "__main__":
    main()
