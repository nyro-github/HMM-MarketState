import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def predict_states():
    # Load Data and Models
    data_path = os.path.join('data', 'processed', 'BTCUSDT_features.csv')
    df = pd.read_csv(data_path, index_col='open_time', parse_dates=True)
    
    # Slice the last 45 days of data to make it much clearer and more granular (45 days * 24 hours = 1080 rows)
    df = df.iloc[-1080:].copy() 
    
    scaler = joblib.load('models/feature_scaler.pkl')
    model = joblib.load('models/market_state_hmm.pkl')
    
    # Scale Features and Predict
    features = ['log_return', 'volatility', 'volume_trend']
    X = df[features].values
    X_scaled = scaler.transform(X)
    
    # Predict the hidden state sequence using the Viterbi algorithm
    df['state'] = model.predict(X_scaled)
    
    return df, model

def plot_market_states(df):
    print("Generating state chart...")
    
    # Define distinct colors for the 8 states
    # You can map these sequentially to meaning later (e.g. State 4 = Red if it is bearish/capitulation)
    colors = ['gray', 'red', 'green', 'blue', 'orange', 'purple', 'black', 'magenta']
    
    # Meaning assigned to each state based on our statistical analysis
    state_names = {
        0: "Accumulation",
        1: "Sharp Downtrend (High Vol)",
        2: "Dead / Low Activity",
        3: "Distribution / Choppy",
        4: "Capitulation (Crash)",
        5: "Explosive Bullish (Parabolic)",
        6: "Steady Bullish",
        7: "Ranging / Vol Spike"
    }
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot the closing price using a simple line
    ax.plot(df.index, df['close'], color='black', alpha=0.6, linewidth=1, label='BTC Price')
    
    # Overlay the states using background colors via axvspan
    # We find contiguous blocks of the same state to color effectively
    df['state_change'] = df['state'].diff().ne(0)
    state_blocks = df[df['state_change']].index
    
    for i in range(len(state_blocks) - 1):
        start = state_blocks[i]
        end = state_blocks[i+1]
        state = df.loc[start, 'state']
        
        ax.axvspan(start, end, color=colors[state], alpha=0.3)
        
    # Handle the very last trailing block
    last_start = state_blocks[-1]
    last_state = df.loc[last_start, 'state']
    ax.axvspan(last_start, df.index[-1], color=colors[last_state], alpha=0.3)
    
    # Formatting
    ax.set_title('BTC Price over 45 Days Colored by HMM Hidden States', fontsize=16)
    ax.set_ylabel('Price (USDT)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_yscale('log') # Log scale is generally better for crypto
    
    # Create custom legend for states
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=colors[i], alpha=0.3, label=f'{i}: {state_names[i]}') for i in range(8)]
    legend_patches.append(plt.Line2D([0], [0], color='black', lw=1, label='Price'))
    ax.legend(handles=legend_patches, loc='upper left')
    
    plt.tight_layout()
    
    # Save the chart
    output_dir = 'visualizations'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'btc_market_states_45d.png'), dpi=300)
    print(f"Chart saved to {output_dir}/btc_market_states_45d.png")
    
    # Show plot interactively if running in a GUI
    # plt.show()

def main():
    print("Loading data and inferring states...")
    df, model = predict_states()
    plot_market_states(df)

if __name__ == "__main__":
    main()
