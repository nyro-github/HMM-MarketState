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

me:

Is there any fundamental problem with our code? It seems like its not really that accurate. Or maybe this is just how HMM is?

copilot gemini3.1pro:

You’ve hit on one of the classic realities of quantitative finance! There is no fundamental "coding bug" in our implementation—what you are seeing is the mathematical reality of Standard Gaussian Hidden Markov Models.

Here are the 4 fundamental reasons why HMMs can feel "inaccurate" or "late" to human eyes, and why it's a known limitation of the model:

1. HMMs inherit the "Lag" of their Features
An HMM does not predict the future; it classifies current data into statistical buckets.

Our Macro model uses the 200 SMA (trend_distance). By definition, moving averages lag behind real-time price action. Therefore, the HMM will not recognize a "Bull Market" transition until the price has already pulled significantly away from the 200 SMA. It isn't catching the exact bottom; it's confirming the trend after it's established.
2. The "Gaussian" Curve vs. Crypto's "Fat Tails"
We are using GaussianHMM. This algorithm mathematically assumes that the market's returns follow a standard "Bell Curve".

The Reality: Crypto does not follow a bell curve. It has massive "fat tails" (sudden violent spikes and deep flash crashes). Because it cannot comprehend fat tails gracefully, the HMM might get confused by a sudden 5% intraday wick and falsely jump into an "Exhaustion Pump" state, only to jump back out an hour later.
3. Humans look at Price; HMMs look at Variance
When you look at a chart, your brain recognizes structural patterns (e.g., Higher Highs, Higher Lows, Head & Shoulders).
The HMM is totally blind to price patterns. It only sees a cocktail of [Returns + Volatility]. To the HMM, a highly volatile bottoming phase looks statistically identical to a highly volatile topping phase because the variance profile is the same, even if the price structure is entirely different.

4. Unsupervised "Labels"
Because this is Unsupervised Learning, the HMM doesn't actually know what a "Bull Market" is. It simply grouped data into 4 statistical clusters (State 0, 1, 2, 3), and we slapped human English labels onto them based on their averages. Sometimes, a specific market state will logically belong in cluster 1 for the HMM, but your human intuition feels it belongs in cluster 2.

How do we make it more accurate?
If you want to move away from these inherent HMM flaws, there are a few architectural pivots we can make:

Swap Lagging features for Leading ones: Stop feeding it Moving Averages. Start feeding it Momentum oscillators (like MACD, RSI) or order-book imbalances if you have them.
Transition to Supervised Learning: If you want it to specifically identify exact setups (like "pre-pump accumulation"), we should move away from HMMs and instead use something like XGBoost or Random Forests, where we explicitly label historical data (e.g., "target variable = a 10% pump happens in the next 24 hours") and train the model to find those exact setups.
Would you prefer to try swapping out the features we feed the HMM to make it more reactive (less lagging), or are you interested in exploring a different type of AI model (like a predictive Supervised model) instead of an Unsupervised clustering)?