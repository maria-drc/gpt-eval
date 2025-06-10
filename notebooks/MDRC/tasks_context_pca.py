import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pytensor.tensor as pt
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates

# 1) DATA PREP FOR MULTI-TASK --------------------------------
def prepare_data_multi(df, task_ids, handle_nan='ignore'):

    df2 = df[df['task_id'].isin(task_ids)][['task_id','Publication date','score']].copy()
    if handle_nan == 'zero':
        df2['score'] = df2['score'].fillna(0)
    else:
        df2 = df2.dropna(subset=['score'])
    df2['Publication date'] = pd.to_datetime(df2['Publication date'])
    df2 = df2.sort_values('Publication date')

    # unified time axis (days since first date)
    unique_dates = sorted(df2['Publication date'].unique())
    first = unique_dates[0]
    time_points = np.array([(d-first).days for d in unique_dates])
    date_to_idx = {d:i for i,d in enumerate(unique_dates)}

    # map each obs to time-index and task-index
    time_idx = df2['Publication date'].map(date_to_idx).values
    task_to_idx = {tid:i for i,tid in enumerate(task_ids)}
    task_idx = df2['task_id'].map(task_to_idx).values

    y_obs = df2['score'].values / 100.0
    return time_points, y_obs, time_idx, task_idx, unique_dates

# 2) MODEL FIT FOR MULTI-TASK --------------------------------
def fit_capability_model_multi(time_points, y_obs, time_idx, task_idx,
                               n_samples=2000, n_tune=1000, random_seed=42):
    T = len(time_points)
    K = len(np.unique(task_idx))

    with pm.Model() as model:
        # Priors
        c0 = pm.Normal('c0', mu=-2.0, sigma=1.0)
        mu = pm.Normal('mu', mu=0.02, sigma=0.01)
        sigma_omega = pm.HalfCauchy('sigma_omega', beta=0.1)
        g = pm.HalfNormal('g', sigma=1.0, shape=K)
        sigma_nu = pm.HalfCauchy('sigma_nu', beta=0.1, shape=K)

        # Latent random walk
        caps = [c0]
        for t in range(1, T):
            dt = time_points[t] - time_points[t-1]
            drift = mu * dt
            eps = pm.Normal(f'eps_{t}', mu=0, sigma=sigma_omega * np.sqrt(dt))
            caps.append(caps[-1] + drift + eps)
        c_stack = pm.Deterministic('c', pt.stack(caps))

        # Fitted mean for each observation
        score_mean = pm.Deterministic(
            'score_mean',
            pm.math.invlogit(g[task_idx] * c_stack[time_idx])
        )

        # Observation
        pm.Normal('y', mu=score_mean, sigma=sigma_nu[task_idx], observed=y_obs)

        trace = pm.sample(
            n_samples, tune=n_tune,
            chains=4, cores=1,
            random_seed=random_seed,
            return_inferencedata=True
        )
    return trace

def fit_capability_model_pooled(time_points, y_obs, time_idx,
                                n_samples=2000, n_tune=1000, random_seed=42):
    T = len(time_points)

    with pm.Model() as model:
        # --- PRIORS ---
        c0           = pm.Normal('c0', mu=-2.0, sigma=1.0)
        mu           = pm.Normal('mu', mu=0.02, sigma=0.01)
        sigma_omega  = pm.HalfCauchy('sigma_omega', beta=0.1)
        # now SHARED across all sensors:
        g            = pm.HalfNormal('g', sigma=1.0)
        sigma_nu     = pm.HalfCauchy('sigma_nu', beta=0.1)

        # --- LATENT RANDOM WALK ---
        caps = [c0]
        for t in range(1, T):
            dt   = time_points[t] - time_points[t-1]
            drift = mu * dt
            eps   = pm.Normal(f'eps_{t}', mu=0, sigma=sigma_omega * np.sqrt(dt))
            caps.append(caps[-1] + drift + eps)
        c_stack = pm.Deterministic('c', pt.stack(caps))

        # --- MEASUREMENT MODEL (shared g & sigma_nu) ---
        # For each observation we pick out the latent c at its time‐index:
        c_for_obs   = c_stack[time_idx]
        score_mean  = pm.Deterministic('score_mean',
                                       pm.math.invlogit(g * c_for_obs))

        pm.Normal('y', mu=score_mean,
                        sigma=sigma_nu,
                        observed=y_obs)

        trace = pm.sample(n_samples, tune=n_tune,
                          chains=4, cores=1,
                          random_seed=random_seed,
                          return_inferencedata=True)
    return trace


def simulate_forecast_from_posterior(trace, time_points, H=100, random_seed=42):
    """
    Draw H‐step‐ahead forecasts from your fitted posterior.
    Returns:
      future_times,
      (c_mean, c_lo, c_hi),
      (y_mean, y_lo, y_hi)
    """
    # 1) Flatten chain+draw → sample
    post = trace.posterior.stack(sample=("chain","draw"))
    
    # 2) Extract & transpose so that axis 0 == samples
    c_arr = post["c"].values            # originally (T, S)
    c_samples = c_arr.T                 # now (S, T)
    
    mu_samples    = post["mu"].values   # (S,)
    sigma_samples = post["sigma_omega"].values  # (S,)
    
    g_arr = post["g"].values            # originally (K, S)
    g_samples = g_arr.T                 # now (S, K)
    
    # 3) Quick sanity print
    print("shapes:", 
          "c",       c_samples.shape,
          "mu",      mu_samples.shape,
          "sigma",   sigma_samples.shape,
          "g",       g_samples.shape)
    
    S, T = c_samples.shape
    K    = g_samples.shape[1]
    
    np.random.seed(random_seed)
    c_fore = np.zeros((S, H))
    y_fore = np.zeros((S, H, K))
    c_last = c_samples[:, -1]
    
    for s in range(S):
        c = c_last[s]
        μ = mu_samples[s]
        σ = sigma_samples[s]
        g = g_samples[s]   # shape (K,)
        
        for h in range(H):
            c = c + μ + np.random.normal(0, σ)
            c_fore[s, h]    = c
            y_fore[s, h, :] = 1/(1 + np.exp(-g*c))
    
    # 4) Summarize to mean + 95% CI
    def summarize(arr):
        return (arr.mean(axis=0),
                np.percentile(arr, 2.5, axis=0),
                np.percentile(arr,97.5, axis=0))
    
    c_mean, c_lo, c_hi = summarize(c_fore)
    y_mean, y_lo, y_hi = summarize(y_fore)
    
    # 5) Build future_times axis
    last_t = time_points[-1]
    future_times = np.arange(last_t+1, last_t+1+H)
    
    return future_times, (c_mean, c_lo, c_hi), (y_mean, y_lo, y_hi)

# 4) PLOTTING UTILITIES --------------------------------------
def plot_raw_data_multi(time_points, y_obs, time_idx, task_idx, task_ids, dates):
    plt.figure(figsize=(10,6))
    markers = ['o','s','^','d','x']
    for i,tid in enumerate(task_ids):
        idx = np.where(task_idx==i)[0]
        plt.scatter(time_points[time_idx[idx]], y_obs[idx],
                    marker=markers[i% len(markers)], s=60)#, label=f'Task {tid}')
    plt.xticks(time_points, [d.strftime('%Y-%m') for d in dates],
               rotation=45, ha='right')
    plt.title('Raw Test Scores'); plt.xlabel('Date'); plt.ylabel('Score')
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

def plot_in_sample_fit(trace, time_points, y_obs, time_idx, task_idx, task_ids, dates):
    # latent
    c_da   = trace.posterior['c']
    c_mean = c_da.mean(dim=('chain','draw')).values
    c_lo   = c_da.quantile(0.025, dim=('chain','draw')).values
    c_hi   = c_da.quantile(0.975, dim=('chain','draw')).values
    # fitted scores
    sm     = trace.posterior['score_mean']
    y_mean = sm.mean(dim=('chain','draw')).values
    y_lo   = sm.quantile(0.025, dim=('chain','draw')).values
    y_hi   = sm.quantile(0.975, dim=('chain','draw')).values

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,5))
    ax1.plot(time_points, c_mean, 'b-'); ax1.fill_between(time_points, c_lo, c_hi, color='b', alpha=0.2)
    ax1.set_title('Estimated Capability'); ax1.set_xticks(time_points)
    ax1.set_xticklabels([d.strftime('%Y-%m') for d in dates], rotation=45, ha='right')
    ax1.grid(alpha=0.3)

    markers = ['o','s','^','d','x']
    for i,tid in enumerate(task_ids):
        idx = np.where(task_idx==i)[0]
        t_obs = time_points[time_idx[idx]]
        ax2.scatter(t_obs, y_obs[idx], marker=markers[i% len(markers)], s=60, label=f'Obs {tid}')
        ax2.plot(t_obs, y_mean[idx], 'b-')
        ax2.fill_between(t_obs, y_lo[idx], y_hi[idx], color='b', alpha=0.2)

    ax2.set_title('In-Sample Fit'); ax2.set_xticks(time_points)
    ax2.set_xticklabels([d.strftime('%Y-%m') for d in dates], rotation=45, ha='right')
    ax2.grid(alpha=0.3); ax2.legend(); plt.tight_layout(); plt.show()


def plot_forecast_simulated(trace, time_points, y_obs, time_idx, task_idx,
                            task_ids, dates, future_times, c_stats, y_stats):

    c_mean, c_lo, c_hi = c_stats
    y_mean_f, y_lo_f, y_hi_f = y_stats

    # 1) Extract in‐sample fit for scores
    sm   = trace.posterior['score_mean']
    y_mean_h = sm.mean(dim=('chain','draw')).values       # (N_obs,)
    y_lo_h   = sm.quantile(0.025, dim=('chain','draw')).values
    y_hi_h   = sm.quantile(0.975, dim=('chain','draw')).values

    # 2) Extract historical latent for left panel
    c_da   = trace.posterior['c']
    ch_mean = c_da.mean(dim=('chain','draw')).values
    ch_lo   = c_da.quantile(0.025, dim=('chain','draw')).values
    ch_hi   = c_da.quantile(0.975, dim=('chain','draw')).values

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5))

    # ---- Left: capability history + forecast ----
    ax1.plot(time_points,   ch_mean, 'b-')
    ax1.fill_between(time_points,   ch_lo,   ch_hi,   color='b', alpha=0.2)
    ax1.plot(future_times,  c_mean, 'g-')
    ax1.fill_between(future_times,  c_lo,  c_hi,  color='g', alpha=0.2)
    ax1.axvline(time_points[-1], color='r', ls='--', label='Forecast start')
    ax1.set_title('Capability Forecast', fontsize=14)
    ax1.grid(alpha=0.3); ax1.legend()
    ax1.set_xticks(time_points)
    ax1.set_xticklabels([d.strftime('%Y-%m') for d in dates], rotation=45, ha='right')
    

    # ---- Right: score history + fit + forecast ----
    markers = ['o','s','^','d','x']
    for i, tid in enumerate(task_ids):
        idx = np.where(task_idx==i)[0]
        t_obs = time_points[time_idx[idx]]

        # a) observations
        ax2.scatter(t_obs, y_obs[idx],
                    marker=markers[i% len(markers)], s=60,
                    label=f'Obs {tid}')

        # b) in‐sample fit line + CI
        ax2.plot(t_obs, y_mean_h[idx], 'b-')
        ax2.fill_between(t_obs,
                         y_lo_h[idx],
                         y_hi_h[idx],
                         color='b', alpha=0.2)

        # c) forecast line + CI
        ax2.plot(future_times, y_mean_f[:,i], 'g-')
        ax2.fill_between(future_times,
                         y_lo_f[:,i],
                         y_hi_f[:,i],
                         color='g', alpha=0.2)

    ax2.axvline(time_points[-1], color='r', ls='--')
    ax2.set_title('Score Forecast', fontsize=14)
    ax2.set_xticks(time_points)
    ax2.set_xticklabels([d.strftime('%Y-%m') for d in dates], rotation=45, ha='right')
    ax2.grid(alpha=0.3)
    # ax2.legend()

    plt.tight_layout()
    plt.savefig('../../results/figures/forecast{}.png'.format(str(task_ids)), dpi=300)
    plt.show()


def simulate_forecast_from_pooled_posterior(trace, time_points, K, H=100, random_seed=42):
    """
    Draw H-step-ahead forecasts from a fitted pooled posterior.
    Returns:
      future_times,
      (c_mean, c_lo, c_hi),
      (y_mean, y_lo, y_hi)  # y[..., k] for each of the K tasks
    """
    # 1) Flatten chain+draw -> sample
    post = trace.posterior.stack(sample=("chain", "draw"))

    # 2) Extract & transpose so that axis 0 == samples
    c_arr = post["c"].values            # (T, S)
    c_samples = c_arr.T                  # (S, T)

    mu_samples    = post["mu"].values   # (S,)
    sigma_samples = post["sigma_omega"].values  # (S,)
    g_samples     = post["g"].values    # (S,)

    S, T = c_samples.shape

    np.random.seed(random_seed)
    c_fore = np.zeros((S, H))
    y_fore = np.zeros((S, H, K))
    c_last = c_samples[:, -1]

    # 3) Simulate forward
    for s in range(S):
        c = c_last[s]
        mu = mu_samples[s]
        sigma = sigma_samples[s]
        g = g_samples[s]
        for h in range(H):
            c = c + mu + np.random.normal(0, sigma)
            c_fore[s, h] = c
            y_val = 1 / (1 + np.exp(-g * c))
            # replicate same forecast for each task
            y_fore[s, h, :] = y_val

    # 4) Summarize to mean + 95% CI
    def summarize(arr):
        return (arr.mean(axis=0),
                np.percentile(arr, 2.5, axis=0),
                np.percentile(arr, 97.5, axis=0))

    c_mean, c_lo, c_hi = summarize(c_fore)
    y_mean, y_lo, y_hi = summarize(y_fore)

    # 5) Build future_times axis
    future_times = np.arange(time_points[-1] + 1,
                             time_points[-1] + 1 + H)

    return future_times, (c_mean, c_lo, c_hi), (y_mean, y_lo, y_hi)

def simulate_forecast_from_pooled_posterior(trace, time_points, K, H=100, random_seed=42):
    """
    Draw H-step-ahead forecasts from a fitted pooled posterior.
    Returns:
      future_times,
      (c_mean, c_lo, c_hi),
      (y_mean, y_lo, y_hi)  # y[..., k] for each of the K tasks
    """
    # 1) Flatten chain+draw -> sample
    post = trace.posterior.stack(sample=("chain", "draw"))

    # 2) Extract & transpose so that axis 0 == samples
    c_arr = post["c"].values            # (T, S)
    c_samples = c_arr.T                  # (S, T)

    mu_samples    = post["mu"].values   # (S,)
    sigma_samples = post["sigma_omega"].values  # (S,)
    g_samples     = post["g"].values    # (S,)

    S, T = c_samples.shape

    np.random.seed(random_seed)
    c_fore = np.zeros((S, H))
    y_fore = np.zeros((S, H, K))
    c_last = c_samples[:, -1]

    # 3) Simulate forward
    for s in range(S):
        c = c_last[s]
        mu = mu_samples[s]
        sigma = sigma_samples[s]
        g = g_samples[s]
        for h in range(H):
            c = c + mu + np.random.normal(0, sigma)
            c_fore[s, h] = c
            y_val = 1 / (1 + np.exp(-g * c))
            # replicate same forecast for each task
            y_fore[s, h, :] = y_val

    # 4) Summarize to mean + 95% CI
    def summarize(arr):
        return (arr.mean(axis=0),
                np.percentile(arr, 2.5, axis=0),
                np.percentile(arr, 97.5, axis=0))

    c_mean, c_lo, c_hi = summarize(c_fore)
    y_mean, y_lo, y_hi = summarize(y_fore)

    # 5) Build future_times axis
    future_times = np.arange(time_points[-1] + 1,
                             time_points[-1] + 1 + H)

    return future_times, (c_mean, c_lo, c_hi), (y_mean, y_lo, y_hi)

def plot_forecast_simulated_pooled(trace, time_points, y_obs, time_idx,
                                   task_idx, task_ids, dates,
                                   future_times, c_stats, y_stats, fig_name='forecast', fig_title=''):
    """
    Plot capability history and score forecasts for the pooled model,
    with plots stacked vertically and capability plot at half the height.
    """
    
    # Custom color palette
    COLOR_PALETTE = ['#FFBD59', '#38B6FF', '#8E3B46', '#E0777D', '#739E82']
    
    c_mean, c_lo, c_hi = c_stats
    y_mean_f, y_lo_f, y_hi_f = y_stats

    # In-sample fit for scores
    sm = trace.posterior['score_mean']
    y_mean_h = sm.mean(dim=('chain', 'draw')).values
    y_lo_h   = sm.quantile(0.025, dim=('chain', 'draw')).values
    y_hi_h   = sm.quantile(0.975, dim=('chain', 'draw')).values

    # Historical latent
    c_da     = trace.posterior['c']
    ch_mean  = c_da.mean(dim=('chain', 'draw')).values
    ch_lo    = c_da.quantile(0.025, dim=('chain', 'draw')).values
    ch_hi    = c_da.quantile(0.975, dim=('chain', 'draw')).values
    
    # Convert integer time_points to actual dates for x-axis
    hist_dates = pd.to_datetime(dates)
    # Generate future dates based on the last historical date
    last_date = hist_dates[-1]
    future_dates = [last_date + pd.Timedelta(days=int(d)) for d in future_times - time_points[-1]]
    
    # Combine historical and future dates for plotting
    all_dates = hist_dates.tolist() + future_dates
    
    # Create figure with vertically stacked subplots with different heights
    fig = plt.figure(figsize=(10, 8))
    # Grid spec with 3 rows (1 for capability, 2 for score forecast)
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 2, 0.1])
    
    # Top panel (capability) - half height
    ax1 = fig.add_subplot(gs[0, 0])
    # Bottom panel (score forecast) - full height
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Despine axes
    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Set up date locator and formatter for both axes
    locator = mdates.MonthLocator(interval=3)  # Every 3 months
    formatter = mdates.DateFormatter('%Y-%m')
    
    # Top panel: capability history + forecast
    ax1.plot(hist_dates, ch_mean, color=COLOR_PALETTE[0], linewidth=2)
    ax1.fill_between(hist_dates, ch_lo, ch_hi, color=COLOR_PALETTE[0], alpha=0.3)
    ax1.plot(future_dates, c_mean, color=COLOR_PALETTE[1], linewidth=2)
    ax1.fill_between(future_dates, c_lo, c_hi, color=COLOR_PALETTE[1], alpha=0.3)
    ax1.axvline(hist_dates[-1], color=COLOR_PALETTE[2], ls='--', label='Forecast start')
    ax1.set_title('Capability Forecast ' + fig_title, fontsize=16)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)
    for label in ax1.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    # ax1.grid(alpha=0.2)
    
    # Bottom panel: score history + forecast
    markers = ['o', 's', '^', 'd', 'x']
    # Scatter observations
    for i, tid in enumerate(task_ids):
        idx = np.where(task_idx == i)[0]
        obs_dates = [hist_dates[time_idx[j]] for j in idx]
        ax2.scatter(obs_dates, y_obs[idx], marker=markers[i % len(markers)],
                    s=40, alpha=0.3)

    # Calculate model fit mean + CI
    unique_dates = sorted(set([hist_dates[ti] for ti in time_idx]))
    fit_avg_y, fit_lo_y, fit_hi_y = [], [], []
    for d in unique_dates:
        sel = [j for j, ti in enumerate(time_idx) if hist_dates[ti] == d]
        if sel:
            vals = y_mean_h[sel]
            fit_avg_y.append(vals.mean())
            fit_lo_y.append(y_lo_h[sel].min())
            fit_hi_y.append(y_hi_h[sel].max())
        else:
            fit_avg_y.append(np.nan)
            fit_lo_y.append(np.nan)
            fit_hi_y.append(np.nan)
    
    # Calculate actual data mean (real mean)
    real_avg_y = []
    for d in unique_dates:
        sel = [j for j, ti in enumerate(time_idx) if hist_dates[ti] == d]
        if sel:
            real_vals = [y_obs[j] for j in sel]
            real_avg_y.append(np.mean(real_vals))
        else:
            real_avg_y.append(np.nan)
            
    # Plot model fit mean
    ax2.plot(unique_dates, fit_avg_y, color=COLOR_PALETTE[0], ls='--', 
             linewidth=1.5, label='Model fit mean')
    ax2.fill_between(unique_dates, fit_lo_y, fit_hi_y, color=COLOR_PALETTE[0], alpha=0.3)
    
    # Plot actual data mean
    ax2.plot(unique_dates, real_avg_y, color=COLOR_PALETTE[3], ls='-', 
             linewidth=2, label='Actual data mean')
    
    # Forecast line for task 0
    ax2.plot(future_dates, y_mean_f[:, 0], color=COLOR_PALETTE[1], linewidth=1)
    ax2.fill_between(future_dates, y_lo_f[:, 0], y_hi_f[:, 0], color=COLOR_PALETTE[1], alpha=0.3)

    ax2.axvline(hist_dates[-1], color=COLOR_PALETTE[2], ls='--')
    ax2.set_title('Score Forecast '  + fig_title, fontsize=16)
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(formatter)
    for label in ax2.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')

    for ax in (ax1, ax2):
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    ax1.set_ylabel('Capability', fontsize=14)
    ax2.set_ylabel('Test scores', fontsize=14)
    # Create legend in the third row
    legend_ax = fig.add_subplot(gs[2, 0])
    legend_ax.axis('off')
    handles, labels = ax2.get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc='center', ncol=3, frameon=False, fontsize=14)
    
    plt.tight_layout()
    plt.savefig('../../results/figures/' + fig_name + '.png', dpi=300)
    plt.show()


# —————————————
# 3) RUN IT

# load your data
df = pd.read_csv('../../results/tables/df_model_test_scores.csv')
df.head()

from scipy.stats import pearsonr
df_clean = df.dropna(subset=['exam_length', 'score'])
corr, p_value = pearsonr(df_clean['exam_length'], df_clean['score'])
print("Correlation coefficient:", corr)
print("p-value:", p_value)

plt.scatter(df['exam_length'], df['score'])
plt.xlabel('Exam Length (characters)')
plt.ylabel('Scores for different models')
plt.show()



# VIA task id we can match with exam length


tasks = df['task_id'][df['occupation_group'] == 'management_occupations'].to_list()
# tasks = df['task_id'][df['occupation_group'] == 'business_and_financial_operations_occupations'].to_list()
# tasks = df['task_id'][df['occupation_group'] == 'computer_and_mathematical_occupations'].to_list()
# tasks = [21522]
tasks = [16246]
len(tasks)
# tasks = tasks[:5]

# prepare & plot raw
tp, y_obs, t_idx, task_idx, dates = prepare_data_multi(df, tasks)
tp, y_obs, t_idx, task_idx, dates = prepare_data_multi(df, tasks, handle_nan='zero')
plot_raw_data_multi(tp, y_obs, t_idx, task_idx, tasks, dates)

# fit
# trace = fit_capability_model_multi(tp, y_obs, t_idx, task_idx,
#                                     n_samples=1000, n_tune=1000)

trace = fit_capability_model_pooled(tp, y_obs, t_idx, 
                                    n_samples=1000, n_tune=1000)

# in-sample diagnostics
az.plot_trace(trace); plt.tight_layout(); plt.show()
print(az.summary(trace, var_names=['mu','sigma_omega','g']))

# plot_in_sample_fit(trace, tp, y_obs, t_idx, task_idx, tasks, dates)


future_times, c_stats, y_stats = simulate_forecast_from_pooled_posterior(
    trace, tp, K=len(tasks), H=150)

plot_forecast_simulated_pooled(
    trace, tp, y_obs, t_idx, task_idx, tasks, dates,
    future_times, c_stats, y_stats
)


plot_forecast_simulated_pooled(
    trace, tp, y_obs, t_idx, task_idx, tasks, dates,
    future_times, c_stats, y_stats, fig_name='forecast_task' + str(tasks), fig_title='Task ' + str(tasks)
)

#### run for different task groups


# Run for different task groups
for occ in df['occupation_group'].unique().tolist():
    print(f"Processing occupation group: {occ}")
    tasks = df['task_id'][df['occupation_group'] == occ].tolist()    
    # Handle NaN as zero
    print(f"Preparing data with handle_nan='zero'")
    tp, y_obs, t_idx, task_idx, dates = prepare_data_multi(df, tasks, handle_nan='zero')
    
    # Fit the model
    print(f"Fitting capability model for {occ}")
    try:
        trace = fit_capability_model_pooled(tp, y_obs, t_idx, 
                                           n_samples=1000, n_tune=1000)
        
        # Generate forecast
        future_times, c_stats, y_stats = simulate_forecast_from_pooled_posterior(
            trace, tp, K=len(tasks), H=150)
        
        # Create plot with NaN handled as zero
        plot_forecast_simulated_pooled(
            trace, tp, y_obs, t_idx, task_idx, tasks, dates,
            future_times, c_stats, y_stats, 
            fig_name=f'forecast_pooled_na0_{occ.replace(" ", "_")}', 
            fig_title=f'Occupation: {occ.replace("_", " ")}'
        )
        
        # Print summary
        print(az.summary(trace, var_names=['mu', 'sigma_omega', 'g']))
    except Exception as e:
        print(f"Error fitting model for {occ} with handle_nan='zero': {e}")
    
    # # Now handle NaN by ignoring
    # print(f"Preparing data with handle_nan='ignore'")
    # try:
    #     tp, y_obs, t_idx, task_idx, dates = prepare_data_multi(df, tasks, handle_nan='ignore')
                
    #     # Fit the model
    #     print(f"Fitting capability model for {occ} ")
    #     trace = fit_capability_model_pooled(tp, y_obs, t_idx, 
    #                                        n_samples=1000, n_tune=1000)
    #     az.plot_trace(trace); plt.tight_layout(); plt.show()
    #     # Generate forecast
    #     future_times, c_stats, y_stats = simulate_forecast_from_pooled_posterior(
    #         trace, tp, K=len(tasks), H=150)
        
    #     # Create plot with NaN handled by ignoring
    #     plot_forecast_simulated_pooled(
    #         trace, tp, y_obs, t_idx, task_idx, tasks, dates,
    #         future_times, c_stats, y_stats, 
    #         fig_name=f'forecast_pooled_naig_{occ.replace(" ", "_")}', 
    #         fig_title=f'Occupation: {occ.replace("_", " ")}'
    #     )
        
    #     # Print summary
    #     print(az.summary(trace, var_names=['mu', 'sigma_omega', 'g']))
    # except Exception as e:
    #     print(f"Error fitting model for {occ} with handle_nan='ignore': {e}")
    
    print(f"Completed processing for {occ}\n")


    