# Forecast simulation using historical returns/volatility; optional test-period verification by comparing median forecast to an actual target price.

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
import argparse

# --- CONFIGURATION & ARGUMENTS ---
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description="Level 36: Asymptotic Target Simulation")
    
    # 1. End Date: The Simulation Start Date (Historical Data Cutoff)
    # Default is 2024-12-09 (Backtest), but can be overridden for live forecasting
    # To simulate a live forecast, replace
    # default="2024-12-09",
    # with
    # default=datetime.now().strftime("%Y-%m-%d"),
    parser.add_argument('--end', type=str, default="2024-12-09",
                        help="Simulation start date (YYYY-MM-DD). Data strictly cut off here.")
    
    # 2. Simulation Parameters
    parser.add_argument('--days', type=int, default=252, help="Trading days to simulate.")
    parser.add_argument('--sims', type=int, default=20000, help="Monte Carlo paths.")
    
    # 3. Verification Price (Result to check against)
    # Default is set to the actual FTSE 100 price on Dec 9, 2025 (approx)
    # This is a backtest, therfore default=9642.01,
    # For live forecast, change to default=None,
    parser.add_argument('--actual-price', type=float, default=9642.01,
                        help="Actual historical price for verification.")
    
    return parser.parse_args()

# Parse arguments
args = get_args()

BACKTEST_END = args.end
NUM_SIMS = args.sims
DAYS = args.days
ACTUAL_FTSE_PRICE = args.actual_price

if ACTUAL_FTSE_PRICE is not None:
    MODE = "BACKTEST"
    print(f"--- RUNNING BACKTEST (Verifying against {ACTUAL_FTSE_PRICE}) ---")
else:
    MODE = "FORECAST"
    print(f"--- RUNNING LIVE FORECAST (No verification price provided) ---")

print(f"Simulation Start: {BACKTEST_END}")
print(f"Horizon:          {DAYS} Days")

TICKERS = ["^FTSE", "^GSPC", "^VIX", "^TNX", "GLD", "CL=F", "GBPUSD=X"] 
START = "2005-01-01" 
INIT_TARGETS = {"Bear": 6600, "Base": 8200, "Bull": 9800}

try:
    print("\n--- 1. LOADING DATA ---")
    
    # --- FIX 1: Prevent Date Ambiguity/Leakage ---
    # We download slightly PAST the target date to ensure the API doesn't exclude our target day.
    # Then we strictly slice the dataframe.
    download_buffer_date = (pd.to_datetime(BACKTEST_END) + timedelta(days=5)).strftime("%Y-%m-%d")
    print(f"Downloading buffer data until {download_buffer_date}...")
    
    raw = yf.download(TICKERS, start=START, end=download_buffer_date, progress=False, auto_adjust=True)
    
    if isinstance(raw.columns, pd.MultiIndex):
        try: 
            data_close = raw["Close"]
            data_vol = raw["Volume"]
        except KeyError: 
            data_close = raw["Adj Close"]
            data_vol = raw["Volume"]
    else:
        data_close = raw["Adj Close"]
        data_vol = raw["Volume"]

    # Handle Negative Oil Prices (April 2020)
    data_close = data_close.where(data_close > 0, np.nan)
    data_close = data_close.ffill().bfill().dropna()
    data_vol = data_vol.ffill().bfill().dropna()
    
    common_index = data_close.index.intersection(data_vol.index)
    data_close = data_close.loc[common_index]
    data_vol = data_vol.loc[common_index]
    
    # STRICT SLICE: Discard anything after the user-defined BACKTEST_END
    # This guarantees no "future data" leaks into the training set.
    cutoff_dt = pd.to_datetime(BACKTEST_END)
    data_close = data_close.loc[data_close.index <= cutoff_dt]
    data_vol = data_vol.loc[data_vol.index <= cutoff_dt]
    
    for t in TICKERS:
        if t not in data_close.columns: raise ValueError(f"Missing {t}")

    print(f"-> Data Loaded: {len(data_close)} days (Ended exactly on {data_close.index[-1].date()}).")
    returns = 100 * np.log(data_close / data_close.shift(1)).dropna()

    # --- 2. TRAINING ASYMPTOTIC ENGINE ---
    print("\n--- 2. TRAINING ASYMPTOTIC ENGINE (LEVEL 36) ---")
    
    hist_vol = returns.ewm(span=20).std().bfill()
    rolling_mean = returns.ewm(span=20).mean()
    std_shocks = (returns - rolling_mean) / hist_vol
    std_shocks = std_shocks.clip(-5, 5).replace([np.inf, -np.inf], 0).dropna()
    
    aligned_index = std_shocks.index
    
    feat_vix = data_close["^VIX"].loc[aligned_index].values
    feat_mom = data_close["^GSPC"].pct_change(50).loc[aligned_index].values * 100
    feat_tnx = data_close["^TNX"].diff(20).loc[aligned_index].fillna(0).values
    feat_oil = data_close["CL=F"].pct_change(20).loc[aligned_index].fillna(0).values * 100
    feat_fx  = data_close["GBPUSD=X"].pct_change(20).loc[aligned_index].fillna(0).values * 100
    
    gspc_vol = data_vol["^GSPC"].loc[aligned_index]
    vol_ma = gspc_vol.rolling(50).mean().replace(0, np.nan)
    feat_vol = (gspc_vol / vol_ma).fillna(1.0).values
    
    features = np.column_stack((feat_vix, feat_mom, feat_tnx, feat_oil, feat_fx, feat_vol))
    features = np.nan_to_num(features, posinf=0.0, neginf=0.0)
    
    cluster_scaler = StandardScaler()
    features_scaled = cluster_scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    library = {}
    for i in range(12):
        subset = std_shocks.values[cluster_labels == i]
        if len(subset) < 10: subset = std_shocks.values 
        library[i] = subset

    mask = np.isfinite(features).all(axis=1)
    vix_med = np.median(feat_vix)
    is_stress = ((feat_vix > vix_med) & (feat_mom < 0)).astype(int)
    
    nn_scaler = StandardScaler()
    X_nn = nn_scaler.fit_transform(features)
    lookback = 1500
    
    nn_model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='tanh', random_state=42, max_iter=1000)
    nn_model.fit(X_nn[-lookback:], is_stress[-lookback:])
    print(f"-> Agent Selector Accuracy: {nn_model.score(X_nn[-lookback:], is_stress[-lookback:]):.2%}")

    # --- 3. RUNNING SIMULATION ---
    print(f"\n--- 3. RUNNING SIMULATION ({DAYS} Days) ---")
    np.random.seed(42)
    
    ftse_idx = list(data_close.columns).index("^FTSE")
    gspc_idx = list(data_close.columns).index("^GSPC")
    vix_idx = list(data_close.columns).index("^VIX")
    tnx_idx = list(data_close.columns).index("^TNX")
    oil_idx = list(data_close.columns).index("CL=F")
    fx_idx  = list(data_close.columns).index("GBPUSD=X")
    
    paths = np.zeros((DAYS + 1, NUM_SIMS, len(TICKERS)))
    paths[0] = data_close.iloc[-1].values
    current_ftse_price = paths[0, 0, ftse_idx]
    
    curr_vols = np.full((NUM_SIMS, len(TICKERS)), hist_vol.iloc[-1].values)
    vol_alpha = 2.0 / (20.0 + 1.0)
    curr_ema = np.full(NUM_SIMS, data_close["^GSPC"].rolling(50).mean().iloc[-1])
    ema_alpha = 0.04
    
    yield_baseline = np.full(NUM_SIMS, data_close["^TNX"].rolling(252).mean().iloc[-1])
    oil_baseline = np.full(NUM_SIMS, data_close["CL=F"].rolling(252).mean().iloc[-1])
    fx_baseline = np.full(NUM_SIMS, data_close["GBPUSD=X"].rolling(252).mean().iloc[-1])
    adapt_rate = 1.0 / 126.0 
    
    start_oil_beta = 0.25
    start_fx_beta = -0.70
    
    for t in range(1, DAYS + 1):
        f_vix = paths[t-1, :, vix_idx]
        safe_ema = np.maximum(curr_ema, 0.01)
        f_mom = (paths[t-1, :, gspc_idx] / safe_ema - 1.0) * 100
        
        if t > 20:
            f_tnx = paths[t-1, :, tnx_idx] - paths[t-21, :, tnx_idx]
            f_oil = (paths[t-1, :, oil_idx] / paths[t-21, :, oil_idx] - 1.0) * 100
            f_fx  = (paths[t-1, :, fx_idx] / paths[t-21, :, fx_idx] - 1.0) * 100
        else:
            f_tnx = np.zeros(NUM_SIMS)
            f_oil = np.zeros(NUM_SIMS)
            f_fx  = np.zeros(NUM_SIMS)
        f_vol = np.ones(NUM_SIMS)
        
        sim_feat = np.column_stack((f_vix, f_mom, f_tnx, f_oil, f_fx, f_vol))
        sim_feat = np.nan_to_num(sim_feat, posinf=0, neginf=0)
        
        sim_feat_cluster = cluster_scaler.transform(sim_feat)
        sim_labels = kmeans.predict(sim_feat_cluster)
        
        sim_feat_nn = nn_scaler.transform(sim_feat)
        prob_irrational = nn_model.predict_proba(sim_feat_nn)[:, 1]
        
        oil_beta = start_oil_beta + (prob_irrational * 0.3)
        fx_beta = start_fx_beta + (prob_irrational * 0.9)
        
        curr_yield = paths[t-1, :, tnx_idx]
        rate_diff = curr_yield - yield_baseline
        rate_impact = (rate_diff / 100.0 * 7.5) 
        yield_factor = np.maximum(0.5, 1.0 - rate_impact)
        
        curr_oil = paths[t-1, :, oil_idx]
        oil_diff = (curr_oil / oil_baseline) - 1.0
        oil_factor = 1.0 + (oil_diff * oil_beta)
        
        curr_fx = paths[t-1, :, fx_idx]
        fx_diff = (curr_fx / fx_baseline) - 1.0
        fx_factor = 1.0 + (fx_diff * fx_beta)
        
        raw_target = INIT_TARGETS["Bull"] * yield_factor * oil_factor * fx_factor
        
        current_price = paths[t-1, :, ftse_idx]
        fomo_factor = np.maximum(1.0, current_price / raw_target)
        fomo_factor = np.minimum(fomo_factor, 1.20)
        fund_target = raw_target * fomo_factor
        
        proximity = current_price / fund_target
        brake_factor = 1.0 - ((proximity - 0.92) * 12.5) 
        brake_factor = np.clip(brake_factor, -0.5, 1.0)
        
        safe_path = np.maximum(current_price, 0.01)
        req_drift = np.log(fund_target / safe_path) / (DAYS - t + 1)
        req_drift = np.clip(req_drift, -0.01, 0.01)
        
        eff_fund_drift = req_drift * brake_factor
        
        mom_drift = (f_mom / 100.0) / 60.0 
        is_pos_gamma = (paths[t-1, :, gspc_idx] > curr_ema)
        is_squeeze = is_pos_gamma & (f_mom > 4.0)
        
        mom_limit = np.where(is_squeeze, 0.01, 0.003)
        mom_drift = np.clip(mom_drift, -mom_limit, mom_limit)
        
        rationality = 1.0 - prob_irrational
        eff_rationality = np.where(is_squeeze, 0.0, rationality)
        
        final_drift = (eff_rationality * eff_fund_drift) + ((1.0 - eff_rationality) * mom_drift)
        
        daily_shocks = np.zeros((NUM_SIMS, len(TICKERS)))
        for c in range(12):
            mask = (sim_labels == c)
            if mask.sum() > 0:
                cluster_lib = library[c]
                ridx = np.random.randint(0, len(cluster_lib), mask.sum())
                daily_shocks[mask] = cluster_lib[ridx]
        
        scaled_returns = daily_shocks * curr_vols
        
        gamma_mult = np.ones(NUM_SIMS)
        gamma_mult[~is_pos_gamma] = 1.4 
        gamma_mult[is_pos_gamma] = 0.7  
        gamma_mult[is_squeeze] = 1.2    
        
        scaled_returns[:, ftse_idx] *= gamma_mult
        scaled_returns[:, gspc_idx] *= gamma_mult
        
        panic_mask = (sim_labels >= 10) & (f_vix > 30)
        if np.any(panic_mask):
            scaled_returns[panic_mask, ftse_idx] *= 1.5
            
        scaled_returns[:, ftse_idx] += (final_drift * 100)
        paths[t] = paths[t-1] * np.exp(scaled_returns / 100)
        
        drawdown_gspc = (paths[t-1, :, gspc_idx] / paths[0, 0, gspc_idx]) - 1.0
        fed_put = (drawdown_gspc < -0.15)
        paths[t, fed_put, tnx_idx] -= 0.05
        
        realized_sq = (scaled_returns / 100)**2 
        curr_vols = np.sqrt(vol_alpha * realized_sq + (1 - vol_alpha) * curr_vols**2)
        
        curr_ema = (paths[t, :, gspc_idx] * ema_alpha) + (curr_ema * (1-ema_alpha))
        paths[t, :, vix_idx] = np.maximum(paths[t, :, vix_idx], 9.0)
        
        yield_baseline = (paths[t, :, tnx_idx] * adapt_rate) + (yield_baseline * (1 - adapt_rate))
        oil_baseline = (paths[t, :, oil_idx] * adapt_rate) + (oil_baseline * (1 - adapt_rate))
        fx_baseline = (paths[t, :, fx_idx] * adapt_rate) + (fx_baseline * (1 - adapt_rate))

    # --- 4. RESULTS ---
    final_ftse = paths[-1, :, ftse_idx]
    median = np.median(final_ftse)
    percent_change = ((median - current_ftse_price) / current_ftse_price) * 100
    
    print(f"\n--- LEVEL 36: ASYMPTOTIC TARGET MODEL ---")
    print(f"Simulation Start:  {BACKTEST_END}")
    print(f"Start Price:       {current_ftse_price:.2f}")
    print(f"Projected Median:  {median:.2f}")
    print(f"Projected Return:  {percent_change:+.2f}%")
    
    if MODE == "BACKTEST" and ACTUAL_FTSE_PRICE is not None:
        error = abs(median - ACTUAL_FTSE_PRICE) / ACTUAL_FTSE_PRICE * 100
        print(f"Actual FTSE (End): {ACTUAL_FTSE_PRICE}")
        print(f"Error Margin:      {error:.2f}%")
    
    plt.figure(figsize=(10, 6))
    plt.hist(final_ftse, bins=100, density=True, color='crimson', alpha=0.5, label='Projected Distribution')
    plt.axvline(current_ftse_price, color='black', linewidth=3, label=f'Start Price ({current_ftse_price:.0f})')
    plt.axvline(median, color='gold', linestyle='--', linewidth=2, label=f'Projected Median ({median:.0f})')
    
    if MODE == "BACKTEST" and ACTUAL_FTSE_PRICE is not None:
         plt.axvline(ACTUAL_FTSE_PRICE, color='red', linewidth=3, label='Actual Price')

    plt.title(f"Level 36: Simulation ({MODE})")
    plt.legend()
    plt.savefig("level36_forecast.png")
    print("Graph saved to 'level36_forecast.png'")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\nCRITICAL ERROR: {e}")