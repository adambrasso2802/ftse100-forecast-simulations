# Forecast simulation using historical returns/volatility; optional test-period verification by comparing median forecast to an actual target price.

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import warnings
import argparse

# --- CONFIGURATION & ARGUMENTS ---
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description="Level 60: FTSE 100 Ensemble Simulation")
    
    # 1. End Date: The Simulation Start Date (Historical Data Cutoff)
    # Default is 2024-12-09 (Backtest), but can be overridden for live forecasting
    # To simulate a live forecast, replace
    # default="2024-12-09",
    # with
    # default=datetime.now().strftime("%Y-%m-%d"),
    parser.add_argument('--end', type=str, default="2024-12-09",
                        help="Simulation start date (YYYY-MM-DD).")
    
    # 2. Simulation Parameters
    parser.add_argument('--days', type=int, default=252, help="Trading days to simulate.")
    parser.add_argument('--sims', type=int, default=20000, help="Monte Carlo paths.")
    
    # 3. Verification Price (Result to check against)
    # Default is set to the actual FTSE 100 price on Dec 9, 2025 (approx)
    # This is a backtest, therfore default=9642.01,
    # For live forecast, change to default=None,
    parser.add_argument('--actual-price', type=float, default=9642.01,
                        help="Actual historical price for verification.")

    # 4. Reproducibility (FIXED)
    parser.add_argument('--seed', type=int, default=42, 
                        help="Random seed for reproducibility (Applied before training).")
    
    return parser.parse_args()

# Parse arguments
args = get_args()

BACKTEST_END = args.end
NUM_SIMS = args.sims
DAYS = args.days
ACTUAL_FTSE_PRICE = args.actual_price
SEED = args.seed

# --- FIX 2: GLOBAL SEEDING ---
# Apply seed immediately so Training (Bootstrap) is reproducible
np.random.seed(SEED)
print(f"--- GLOBAL SEED SET TO: {SEED} ---")

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
N_ENSEMBLE = 10 
INIT_TARGETS = {"Bear": 6600, "Base": 8200, "Bull": 9800}

try:
    print("\n--- 1. LOADING DATA ---")
    
    # --- FIX 1: Prevent Date Ambiguity/Leakage ---
    # Download buffer
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

    # Handle Negative Oil Prices
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

    # --- 2. TRAINING ENSEMBLE REGRESSION (LEVEL 60) ---
    print("\n--- 2. TRAINING ENSEMBLE REGRESSION (LEVEL 60) ---")
    
    reg_window = 500
    df_reg = data_close.iloc[-reg_window:].copy()
    y_reg = np.log(df_reg["^FTSE"])
    X_reg = pd.DataFrame({
        'GSPC': np.log(df_reg["^GSPC"]),
        'OIL': np.log(df_reg["CL=F"]),
        'FX': np.log(df_reg["GBPUSD=X"]),
        'TNX': df_reg["^TNX"]
    })
    
    ecm_coefs_ensemble = []
    fair_value_estimates = []
    
    last_features = X_reg.iloc[[-1]] 
    
    for i in range(N_ENSEMBLE):
        # This sampling is now deterministic because np.random.seed was set at the top
        sample_size = int(reg_window * 0.8)
        sample_indices = np.random.choice(reg_window, size=sample_size, replace=True)
        
        X_sample = X_reg.iloc[sample_indices]
        y_sample = y_reg.iloc[sample_indices]
        
        ecm_model = LinearRegression()
        ecm_model.fit(X_sample, y_sample)
        
        ecm_coefs_ensemble.append({
            'intercept': ecm_model.intercept_,
            'coefs': ecm_model.coef_
        })
        
        pred = ecm_model.predict(last_features)[0]
        fair_value_estimates.append(pred)

    print(f"-> Trained {N_ENSEMBLE} independent ECM models.")
    fair_value_last = np.mean(fair_value_estimates) 
    
    actual_value_last = y_reg.iloc[-1]
    initial_alpha_bias = actual_value_last - fair_value_last
    
    alpha_floor = initial_alpha_bias * 0.5 
    ALPHA_DECAY = 0.995 
    
    # B. AGENT SELECTOR
    gspc_vol = data_vol["^GSPC"].loc[returns.index]
    vol_ma = gspc_vol.rolling(50).mean().replace(0, np.nan)
    rel_volume = (gspc_vol / vol_ma).fillna(1.0)
    
    feat_vix = data_close["^VIX"].loc[returns.index].values
    feat_mom = data_close["^GSPC"].pct_change(50).loc[returns.index].values * 100
    feat_tnx = data_close["^TNX"].diff(20).loc[returns.index].fillna(0).values
    feat_oil = data_close["CL=F"].pct_change(20).loc[returns.index].fillna(0).values * 100
    feat_fx  = data_close["GBPUSD=X"].pct_change(20).loc[returns.index].fillna(0).values * 100
    feat_vol = rel_volume.values
    
    features = np.column_stack((feat_vix, feat_mom, feat_tnx, feat_oil, feat_fx, feat_vol))
    features = np.nan_to_num(features, posinf=0.0, neginf=0.0)
    
    cluster_scaler = StandardScaler()
    features_scaled = cluster_scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=12, random_state=SEED, n_init=10) # Seeded
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    hist_vol = returns.ewm(span=20).std().bfill()
    rolling_mean = returns.ewm(span=20).mean()
    std_shocks = (returns - rolling_mean) / hist_vol
    std_shocks = std_shocks.clip(-5, 5).replace([np.inf, -np.inf], 0).dropna()
    
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
    nn_model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='tanh', random_state=SEED, max_iter=1000)
    nn_model.fit(X_nn[-lookback:], is_stress[-lookback:])
    
    # --- 3. RUNNING SIMULATION ---
    print(f"\n--- 3. RUNNING SIMULATION ({DAYS} Steps) ---")
    # np.random.seed is already set globally at start
    
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
    
    q_ends = [63, 126, 189, 252]
    daily_buyback = 0.038 / 252.0 
    target_vol_daily = 12.0 / np.sqrt(252)
    curr_leverage = np.ones(NUM_SIMS) 
    decay_state = np.full(NUM_SIMS, initial_alpha_bias - alpha_floor)
    prev_drift = np.zeros(NUM_SIMS) 
    
    curr_bull_target_static = np.full(NUM_SIMS, float(INIT_TARGETS["Bull"]))
    
    ecm_model_indices = np.random.randint(0, N_ENSEMBLE, NUM_SIMS)
    
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
        rationality = 1.0 - prob_irrational
        
        realized_vol_equity = (curr_vols[:, ftse_idx] + curr_vols[:, gspc_idx]) / 2.0
        safe_realized = np.maximum(realized_vol_equity, 0.5) 
        target_leverage = np.clip(target_vol_daily / safe_realized, 0.2, 2.0)
        delta_leverage = (target_leverage - curr_leverage) * 0.10
        vol_control_flow = delta_leverage * 0.005 
        curr_leverage += delta_leverage
        
        days_to_q_end = np.min([abs(t - q) for q in q_ends])
        is_blackout = days_to_q_end <= 10
        is_front_run = (days_to_q_end > 10) & (days_to_q_end <= 15)
        
        buyback_elasticity = np.where(paths[t-1, :, ftse_idx] < curr_ema, 2.5, 0.5)
        base_flow = daily_buyback * buyback_elasticity
        buyback_flow = np.where(is_blackout, 0.0, base_flow)
        buyback_flow = np.where(is_front_run, base_flow * 2.0, buyback_flow)
        
        rebal_flow = np.zeros(NUM_SIMS)
        if days_to_q_end <= 5: 
            lookback_idx = max(0, t - 63)
            q_return = (paths[t-1, :, ftse_idx] / paths[lookback_idx, :, ftse_idx]) - 1.0
            sell_trigger = (q_return > 0.05) & (f_mom < 2.0)
            rebal_flow[sell_trigger] = -(q_return[sell_trigger] * 0.05)
            buy_trigger = q_return < -0.05
            rebal_flow[buy_trigger] = -(q_return[buy_trigger] * 0.05)
            rebal_flow = np.clip(rebal_flow, -0.005, 0.005)
            
        decay_state *= ALPHA_DECAY
        current_alpha_bias = decay_state + alpha_floor
        
        ecm_intercepts_t = np.array([ecm_coefs_ensemble[i]['intercept'] for i in ecm_model_indices])
        ecm_coefs_t = np.array([ecm_coefs_ensemble[i]['coefs'] for i in ecm_model_indices])
        
        sim_log_gspc = np.log(np.maximum(paths[t-1, :, gspc_idx], 0.01))
        sim_log_oil = np.log(np.maximum(paths[t-1, :, oil_idx], 0.01))
        sim_log_fx = np.log(np.maximum(paths[t-1, :, fx_idx], 0.01))
        sim_tnx = paths[t-1, :, tnx_idx]
        
        fair_log_ftse = ecm_intercepts_t + \
                        (ecm_coefs_t[:, 0] * sim_log_gspc) + \
                        (ecm_coefs_t[:, 1] * sim_log_oil) + \
                        (ecm_coefs_t[:, 2] * sim_log_fx) + \
                        (ecm_coefs_t[:, 3] * sim_tnx) + \
                        current_alpha_bias
                        
        curr_log_ftse = np.log(np.maximum(paths[t-1, :, ftse_idx], 0.01))
        pricing_gap = fair_log_ftse - curr_log_ftse
        
        drift_up = np.maximum(0, pricing_gap) * 0.05
        drift_down = np.minimum(0, pricing_gap) * 0.05
        allow_selling = f_mom < 0 
        drift_down = np.where(allow_selling, drift_down, drift_down * 0.1) 
        ecm_drift = np.clip(drift_up + drift_down, -0.01, 0.01)
        
        current_price = paths[t-1, :, ftse_idx]
        
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
        
        w_bull = np.clip(1.0 - (prob_irrational * 1.5), 0.0, 1.0)
        base_target_mix = (curr_bull_target_static * w_bull) + (INIT_TARGETS["Base"] * (1-w_bull))
        fund_target = base_target_mix * yield_factor * oil_factor * fx_factor
        
        fomo_factor = np.clip(np.maximum(1.0, current_price / fund_target), 1.0, 1.05)
        fund_target = fund_target * fomo_factor
        
        safe_path = np.maximum(current_price, 0.01)
        log_price = np.log(safe_path)
        log_target = np.log(np.maximum(fund_target, 0.01))
        distance = log_target - log_price
        
        base_theta = (rationality * 0.10) + ((1.0 - rationality) * 0.01)
        elastic_theta = np.abs(distance) * 0.5 
        final_theta = base_theta + elastic_theta
        ou_drift = distance * final_theta
        
        mom_drift_base = np.clip((f_mom / 100.0) / 60.0, -0.003, 0.003)
        w_ou = rationality
        w_mom = 1.0 - rationality
        
        sim_vol_decimal = curr_vols[:, ftse_idx] / 100.0
        variance_drag_comp = 0.5 * (sim_vol_decimal ** 2)
        
        calculated_drift = (w_ou * (ou_drift + variance_drag_comp)) + (w_mom * mom_drift_base) + \
                           rebal_flow + buyback_flow + vol_control_flow + ecm_drift
        
        final_drift = (0.8 * calculated_drift) + (0.2 * prev_drift)
        prev_drift = final_drift
        
        daily_shocks = np.zeros((NUM_SIMS, len(TICKERS)))
        for c in range(12):
            mask = (sim_labels == c)
            if mask.sum() > 0:
                cluster_lib = library[c]
                ridx = np.random.randint(0, len(cluster_lib), mask.sum())
                daily_shocks[mask] = cluster_lib[ridx]
        
        scaled_returns = daily_shocks * curr_vols
        
        if t > 1:
            delta_vix = paths[t-1, :, vix_idx] - paths[t-2, :, vix_idx]
        else:
            delta_vix = np.zeros(NUM_SIMS)
        vanna_flow = np.clip(delta_vix * -0.0005, -0.005, 0.005)
        scaled_returns[:, ftse_idx] += (vanna_flow * 100)
        
        gamma_mult = np.ones(NUM_SIMS)
        is_pos_gamma = (paths[t-1, :, gspc_idx] > curr_ema)
        is_squeeze = is_pos_gamma & (f_mom > 3.5)
        
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
        paths[t, drawdown_gspc < -0.15, tnx_idx] -= 0.05
        realized_sq = (scaled_returns / 100)**2 
        curr_vols = np.sqrt(vol_alpha * realized_sq + (1 - vol_alpha) * curr_vols**2)
        
        curr_ema = (paths[t, :, gspc_idx] * ema_alpha) + (curr_ema * (1-ema_alpha))
        paths[t, :, vix_idx] = np.maximum(paths[t, :, vix_idx], 9.0)
        paths[t, :, oil_idx] = np.maximum(paths[t, :, oil_idx], 10.0) 
        
        yield_baseline = (paths[t, :, tnx_idx] * adapt_rate) + (yield_baseline * (1 - adapt_rate))
        oil_baseline = (paths[t, :, oil_idx] * adapt_rate) + (oil_baseline * (1 - adapt_rate))
        fx_baseline = (paths[t, :, fx_idx] * adapt_rate) + (fx_baseline * (1 - adapt_rate))

    # --- 4. RESULTS ---
    final_ftse = paths[-1, :, ftse_idx]
    median = np.median(final_ftse)
    
    print(f"\n--- LEVEL 60: ENSEMBLE CALIBRATION MODEL ---")
    print(f"Simulation Start:  {BACKTEST_END}")
    print(f"Start Price:       {current_ftse_price:.2f}")
    print(f"Projected Median:  {median:.2f}")
    
    percent_change = ((median - current_ftse_price) / current_ftse_price) * 100
    print(f"Projected Return:  {percent_change:+.2f}%")

    if MODE == "BACKTEST" and ACTUAL_FTSE_PRICE is not None:
        error = abs(median - ACTUAL_FTSE_PRICE) / ACTUAL_FTSE_PRICE * 100
        print(f"Actual FTSE (End): {ACTUAL_FTSE_PRICE}")
        print(f"Error Margin:      {error:.2f}%")
    
    plt.figure(figsize=(10, 6))
    plt.hist(final_ftse, bins=100, density=True, color='purple', alpha=0.6, label='Ensemble Dist.')
    plt.axvline(current_ftse_price, color='black', linewidth=3, label=f'Start Price ({current_ftse_price:.0f})')
    plt.axvline(median, color='gold', linestyle='--', linewidth=2, label='Model Median')
    
    if MODE == "BACKTEST" and ACTUAL_FTSE_PRICE is not None:
         plt.axvline(ACTUAL_FTSE_PRICE, color='red', linewidth=3, label='Actual Price')

    plt.title(f"Level 60: Simulation ({MODE})")
    plt.legend()
    plt.savefig("level60_ensemble.png")
    print("Graph saved to 'level60_ensemble.png'")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\nCRITICAL ERROR: {e}")
