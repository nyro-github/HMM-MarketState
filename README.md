# Hidden Markov Model (HMM) for Cryptocurrency Market States

This project implements a Hidden Markov Model (HMM) to identify and predict 8 distinct macroeconomic and microeconomic market states for a specific cryptocurrency.

##  The 8 Market States

Based on typical financial asset behavior involving returns, volatility, and volume, the model aims to classify the market into the following 8 states:

1. **Dead / Low Activity**: Extremely low volatility, minimal trading volume, and near-zero returns.
2. **Accumulation**: Low volatility, flat or slight positive returns, steady volume (smart money buying).
3. **Distribution**: Low to moderate volatility, flat or slight negative returns (smart money selling).
4. **Steady Bullish**: Consistent positive returns with moderate volatility (healthy uptrend).
5. **Explosive Bullish (Parabolic)**: Extreme positive returns, high volatility, and massive volume. 
6. **Steady Bearish**: Consistent negative returns with moderate volatility (sustained downtrend).
7. **Capitulation (Crash)**: Extreme negative returns, massive volatility spikes, panic selling volume.
8. **Choppy / Ranging**: High volatility but mean-reverting near-zero returns (whipsaw market, dangerous for trend followers).

##  Implementation Plan

### Phase 1: Data Collection & Feature Engineering
- **Collect OHLCV Data**: Use an API (like Binance, CCXT, or Yahoo Finance) to fetch historical data for the target coin (e.g., BTC/USDT).
- **Calculate Log Returns**: The primary indicator of price direction.
- **Calculate Realized Volatility**: A rolling standard deviation of log returns.
- **Identify Volume Trends**: Calculate a volume oscillator or moving average ratio for volume.

### Phase 2: Model Architecture
- **Library Selection**: Use Python's hmmlearn library (GaussianHMM or GMMHMM).
- **Configuration**: Set 
_components=8 (for the 8 states).
- **Features Selection**: Feed the model a multi-dimensional array containing: [Log Returns, Volatility, Volume Change].

### Phase 3: Training and Convergence
- **Data Scaling**: Standardize features using StandardScaler to ensure the HMM converges properly.
- **Model Fitting**: Run the Expectation-Maximization (EM) algorithm to fit the model. Since EM is prone to local optima, train multiple models with different random seeds and select the one with the highest log-likelihood.

### Phase 4: State Interpretation
- **Decoding State Sequence**: Use the Viterbi algorithm (model.predict(X)) to infer the most likely market state at each past time step.
- **Mapping States**: HMM assigns arbitrary integers (0-7) to states. We mathematically analyze the mean and variance of each state's returns/volatility to map them to our human-readable labels (e.g., mapping State 4 to "Steady Bullish").

### Phase 5: Visualization & Macro Analysis
- **Plotting**: Overlay the 8 market states on a price chart using distinct background colors.
- **Transition Matrix Analysis**: Analyze the state transition matrix to find the probabilities of moving from one state to another (e.g., what is the probability of moving from *Accumulation* to *Steady Bullish* vs. *Capitulation*).
