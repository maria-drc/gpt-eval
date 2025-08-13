"""
Script used for Vienna presentation to fit a Bayesian Dynamic Factor Model to the test scores of different models on different tasks.
Included also PCA analysis, with and without claude 3.5 sonnet
"""
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pytensor.tensor as pt
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
from scipy.stats import pearsonr
COLOR_PALETTE = ['#FFBD59', '#38B6FF', '#8E3B46', '#E0777D', '#739E82']
    
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

# —————————————
# 3) RUN IT

# load your data
df = pd.read_csv('../../results/tables/df_model_test_scores.csv')
df.head()


df_clean = df.dropna(subset=['exam_length', 'score'])
corr, p_value = pearsonr(df_clean['exam_length'], df_clean['score'])
print("Correlation coefficient with outliers:", corr)
print("p-value:", p_value)

df_clean = df.dropna(subset=['exam_length', 'score'])
corr, p_value = pearsonr(df_clean['exam_length'][df_clean['exam_length'] > 25000], df_clean['score'][df_clean['exam_length'] > 25000])
print("Correlation coefficient without outliers:", corr)
print("p-value:", p_value)


plt.scatter(df['exam_length'], df['score'])
plt.xlabel('Exam Length (characters)')
plt.ylabel('Scores for different models')
plt.show()

# TASKS with highest length
df[['task_id','occupation', 'task_description', 'exam_length']][df['exam_length'] > 25000].head()
pd.set_option('display.max_colwidth', None)

# Select tasks with exam_length > 25000 and print the first three rows
print(df[['task_id', 'occupation', 'task_description']][df['exam_length'] > 25000].head(3))


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


df_task = df.groupby('task_id').agg({
    'exam_length': 'first',
    'score': ['mean', 'std'],
    'occupation_group': 'first'
}).reset_index()
df_task.columns = ['task_id', 'exam_length', 'score_mean', 'score_std', 'occupation_group']

# Define a colors mapping using your provided color palette (assume only three groups)
occ_groups = sorted(df_task['occupation_group'].unique())
COLOR_PALETTE = ['#FFBD59', '#38B6FF', '#8E3B46', '#E0777D', '#739E82']
color_mapping = dict(zip(occ_groups, COLOR_PALETTE[:len(occ_groups)]))

df_task = df.groupby('task_id').agg({
    'exam_length': 'first',
    'score': ['mean', 'std'],
    'occupation_group': 'first'
}).reset_index()
df_task.columns = ['task_id', 'exam_length', 'score_mean', 'score_std', 'occupation_group']

# Define a colors mapping using your provided color palette (assume only three groups)
occ_groups = sorted(df_task['occupation_group'].unique())
COLOR_PALETTE = ['#FFBD59', '#38B6FF', '#8E3B46', '#E0777D', '#739E82']
color_mapping = dict(zip(occ_groups, COLOR_PALETTE[:len(occ_groups)]))

plt.figure(figsize=(10, 6))
ax = plt.gca()
# Plot each occupation group with error bars
for occ in occ_groups:
    sub_df = df_task[df_task['occupation_group'] == occ]
    plt.errorbar(sub_df['exam_length'], sub_df['score_mean'],
                 yerr=sub_df['score_std'],
                 fmt='o', label=occ,
                 ecolor='gray', 
                 color=color_mapping[occ],
                 capsize=2, markersize=10, alpha=0.7)

plt.xlabel('Exam Length (characters)', fontsize=16)
plt.ylabel('Average Score', fontsize=16)
plt.ylabel('Mean task score across models', fontsize=16)
plt.ylim([0,100])
# Despine: remove top and right borders.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('../../results/figures/exam_length_vs_score_errorbars.png', bbox_inches='tight')
plt.show()



plt.figure(figsize=(10, 6))
ax = plt.gca()
# Plot each occupation group with error bars
for occ in occ_groups:
    sub_df = df_task[df_task['occupation_group'] == occ]
    plt.scatter(sub_df['exam_length'], sub_df['score_mean'],
            label=occ,
            color=color_mapping[occ],
            s=150,    # adjust marker size as needed
            alpha=0.7)

plt.xlabel('Exam length (characters)', fontsize=16)
plt.ylabel('Mean task score across models', fontsize=16)
# plt.title('Correlation between Exam Length and Average Score per Task', fontsize=18)
plt.ylim([0,100])
# Despine: remove top and right borders.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('../../results/figures/exam_length_vs_score.png', bbox_inches='tight')
plt.show()


########## Factor analysis on tasks
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pytensor.tensor as pt
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
from scipy.stats import pearsonr
from scipy.special import logit, expit
import warnings
warnings.filterwarnings('ignore')

COLOR_PALETTE = ['#FFBD59', '#38B6FF', '#8E3B46', '#E0777D', '#739E82']
    
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
    # Clip to avoid logit issues
    y_obs = np.clip(y_obs, 0.001, 0.999)
    
    return time_points, y_obs, time_idx, task_idx, unique_dates, task_to_idx

# 2) BAYESIAN DYNAMIC FACTOR MODEL -------------------------
def build_factor_model(y_obs, time_idx, task_idx, n_factors, n_times, n_tasks):
    """
    Build Bayesian Dynamic Factor Model with sigmoid observations
    """
    with pm.Model() as model:
        # Prior hyperparameters
        mu_drift_mean = 0.02  # Expected improvement rate
        mu_drift_sd = 0.01
        
        # Factor drift parameters (positive)
        mu = pm.TruncatedNormal('mu', mu=mu_drift_mean, sigma=mu_drift_sd, 
                               lower=0, shape=(n_factors,))
        
        # Factor innovation variances
        sigma_f = pm.InverseGamma('sigma_f', alpha=2, beta=0.1, shape=(n_factors,))
        
        # Initial factor values (negative for early poor performance)
        f_init = pm.Normal('f_init', mu=-2.0, sigma=1.0, shape=(n_factors,))
        
        # Factor evolution (random walk with drift)
        f_innov = pm.Normal('f_innov', mu=0, sigma=1, shape=(n_factors, n_times-1))
        
        # Build factor trajectories
        f = pt.zeros((n_factors, n_times))
        f = pt.set_subtensor(f[:, 0], f_init)
        
        for t in range(1, n_times):
            f = pt.set_subtensor(f[:, t], 
                               f[:, t-1] + mu + sigma_f * f_innov[:, t-1])
        
        # Factor loadings with lower triangular constraint
        # Only estimate loadings for valid (i,k) pairs where k <= i
        lambda_raw = pm.Normal('lambda_raw', mu=0, sigma=1.0, 
                              shape=(n_tasks * n_factors,))
        
        # Create lower triangular loading matrix
        Lambda = pt.zeros((n_tasks, n_factors))
        idx = 0
        for i in range(n_tasks):
            for k in range(min(i+1, n_factors)):
                if i == k:  # Diagonal elements (positive)
                    Lambda = pt.set_subtensor(Lambda[i, k], 
                                            pt.exp(lambda_raw[idx]))
                else:  # Lower triangular elements
                    Lambda = pt.set_subtensor(Lambda[i, k], lambda_raw[idx])
                idx += 1
        
        # Task-specific capabilities
        c = pt.sum(Lambda[task_idx, :] * f[:, time_idx].T, axis=1)
        
        # Observation noise
        tau = pm.InverseGamma('tau', alpha=2, beta=0.1, shape=(n_tasks,))
        
        # Observation model (logit-normal)
        y_logit = pm.Normal('y_logit', mu=c, sigma=tau[task_idx], 
                           observed=logit(y_obs))
        
        # Store useful quantities
        pm.Deterministic('factors', f)
        pm.Deterministic('loadings', Lambda)
        pm.Deterministic('capabilities', c)
        
    return model

# 3) MODEL FITTING AND SELECTION ----------------------------
def fit_factor_models(y_obs, time_idx, task_idx, n_times, n_tasks, 
                     k_range=[2, 3, 4], n_samples=1000, n_tune=1000):
    """
    Fit models with different numbers of factors and compute WAIC
    """
    models = {}
    traces = {}
    waic_scores = {}
    
    for k in k_range:
        print(f"\nFitting model with {k} factors...")
        
        model = build_factor_model(y_obs, time_idx, task_idx, k, n_times, n_tasks)
        
        with model:
            trace = pm.sample(n_samples, tune=n_tune, 
                            target_accept=0.90, random_seed=42)
            
            # Compute WAIC
            waic = az.waic(trace, model)
            
            models[k] = model
            traces[k] = trace
            waic_scores[k] = waic.waic
            
            print(f"WAIC for K={k}: {waic.waic:.2f}")
    
    # Select best model
    best_k = min(waic_scores.keys(), key=lambda k: waic_scores[k])
    print(f"\nBest model: K={best_k} (WAIC={waic_scores[best_k]:.2f})")
    
    return models, traces, waic_scores, best_k

# 4) ANALYSIS AND INTERPRETATION -------------------------
def analyze_factors(trace, task_ids, task_to_idx, dates, best_k):
    """
    Analyze and interpret factor model results
    """
    # Extract posterior means
    factors = trace.posterior['factors'].mean(dim=['chain', 'draw']).values
    loadings = trace.posterior['loadings'].mean(dim=['chain', 'draw']).values
    mu_posterior = trace.posterior['mu'].mean(dim=['chain', 'draw']).values
    
    # Print factor interpretation
    print(f"\n=== FACTOR ANALYSIS (K={best_k}) ===")
    print("\nFactor drift rates (capability improvement):")
    for k in range(best_k):
        print(f"  Factor {k+1}: {mu_posterior[k]:.4f} per day")
    
    print(f"\nFactor loadings (top 3 tasks per factor):")
    for k in range(best_k):
        loading_ranks = np.argsort(np.abs(loadings[:, k]))[::-1]
        print(f"\n  Factor {k+1}:")
        for i in range(min(3, len(task_ids))):
            task_idx = loading_ranks[i]
            task_id = task_ids[task_idx]
            loading_val = loadings[task_idx, k]
            print(f"    Task {task_id}: {loading_val:.3f}")
    
    return factors, loadings, mu_posterior

# 5) PLOTTING UTILITIES --------------------------------------
def plot_raw_data_multi(time_points, y_obs, time_idx, task_idx, task_ids, dates):
    plt.figure(figsize=(12,8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(task_ids)))
    
    for i, tid in enumerate(task_ids):
        idx = np.where(task_idx == i)[0]
        if len(idx) > 0:
            plt.scatter(time_points[time_idx[idx]], y_obs[idx],
                       color=colors[i], s=60, alpha=0.7, label=f'Task {tid}')
    
    plt.xticks(time_points, [d.strftime('%Y-%m') for d in dates],
               rotation=45, ha='right')
    plt.title('Raw Test Scores by Task')
    plt.xlabel('Publication Date')
    plt.ylabel('Score')
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_factor_evolution(factors, dates, mu_posterior, best_k):
    """
    Plot factor evolution over time
    """
    fig, axes = plt.subplots(1, best_k, figsize=(5*best_k, 4))
    if best_k == 1:
        axes = [axes]
    
    for k in range(best_k):
        axes[k].plot(dates, factors[k, :], 'o-', color=COLOR_PALETTE[k % len(COLOR_PALETTE)])
        axes[k].set_title(f'Factor {k+1}\n(drift: {mu_posterior[k]:.4f}/day)')
        axes[k].set_xlabel('Date')
        axes[k].set_ylabel('Factor Level')
        axes[k].grid(alpha=0.3)
        axes[k].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_factor_loadings(loadings, task_ids, best_k):
    """
    Plot factor loading matrix as heatmap
    """
    plt.figure(figsize=(max(6, best_k*2), max(6, len(task_ids)*0.3)))
    im = plt.imshow(loadings, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im, label='Loading')
    plt.yticks(range(len(task_ids)), [f'Task {tid}' for tid in task_ids])
    plt.xticks(range(best_k), [f'Factor {k+1}' for k in range(best_k)])
    plt.title('Factor Loading Matrix')
    plt.tight_layout()
    plt.show()

# 6) TASK CLUSTERING BASED ON LOADINGS -------------------
def cluster_tasks_by_loadings(loadings, task_ids, n_clusters=3):
    """
    Cluster tasks based on factor loading patterns
    """
    from sklearn.cluster import KMeans
    
    # Cluster tasks based on loadings
    kmeans = KMeans(n_clusters=n_clusters, random_seed=42)
    clusters = kmeans.fit_predict(loadings)
    
    print(f"\n=== TASK CLUSTERING ===")
    for c in range(n_clusters):
        task_indices = np.where(clusters == c)[0]
        cluster_tasks = [task_ids[i] for i in task_indices]
        print(f"\nCluster {c+1}: {cluster_tasks}")
    
    return clusters

# 7) MAIN EXECUTION PIPELINE ------------------------------
def run_factor_analysis(df, task_ids, k_range=[2, 3, 4]):
    """
    Complete pipeline for factor analysis
    """
    print("=== BAYESIAN DYNAMIC FACTOR MODEL ANALYSIS ===\n")
    
    # 1. Prepare data
    print("1. Preparing data...")
    time_points, y_obs, time_idx, task_idx, dates, task_to_idx = prepare_data_multi(df, task_ids)
    n_times = len(time_points)
    n_tasks = len(task_ids)
    n_obs = len(y_obs)
    
    print(f"   Data: {n_obs} observations, {n_tasks} tasks, {n_times} time points")
    
    # 2. Plot raw data
    print("\n2. Plotting raw data...")
    plot_raw_data_multi(time_points, y_obs, time_idx, task_idx, task_ids, dates)
    
    # 3. Fit models
    print("\n3. Fitting factor models...")
    models, traces, waic_scores, best_k = fit_factor_models(
        y_obs, time_idx, task_idx, n_times, n_tasks, k_range)
    
    # 4. Analyze best model
    print("\n4. Analyzing best model...")
    factors, loadings, mu_posterior = analyze_factors(
        traces[best_k], task_ids, task_to_idx, dates, best_k)
    
    # 5. Plot results
    print("\n5. Plotting results...")
    plot_factor_evolution(factors, dates, mu_posterior, best_k)
    plot_factor_loadings(loadings, task_ids, best_k)
    
    # 6. Cluster tasks
    print("\n6. Clustering tasks...")
    clusters = cluster_tasks_by_loadings(loadings, task_ids)
    
    return {
        'models': models,
        'traces': traces, 
        'waic_scores': waic_scores,
        'best_k': best_k,
        'factors': factors,
        'loadings': loadings,
        'clusters': clusters,
        'data': (time_points, y_obs, time_idx, task_idx, dates, task_to_idx)
    }

# Example usage:

# Load your data
df = pd.read_csv('../../results/tables/df_model_test_scores.csv')

# Select subset of tasks for analysis (start small)
available_tasks = df['task_id'].unique()
task_ids = available_tasks[:100]  # Start with 10 tasks

# Run complete analysis
results = run_factor_analysis(df, task_ids, k_range=[2, 3, 4])

print("\n=== ANALYSIS COMPLETE ===")
print(f"Best model has {results['best_k']} factors")
print("Check plots and clustering results above!")

#####

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pytensor.tensor as pt
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
from scipy.stats import pearsonr
from scipy.special import logit, expit
import warnings
warnings.filterwarnings('ignore')

COLOR_PALETTE = ['#FFBD59', '#38B6FF', '#8E3B46', '#E0777D', '#739E82']
    
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
    # Clip to avoid logit issues
    y_obs = np.clip(y_obs, 0.001, 0.999)
    
    return time_points, y_obs, time_idx, task_idx, unique_dates, task_to_idx

# 2) BAYESIAN DYNAMIC FACTOR MODEL -------------------------
def build_factor_model(y_obs, time_idx, task_idx, n_factors, n_times, n_tasks):
    """
    Simplified Bayesian Dynamic Factor Model with sigmoid observations
    """
    with pm.Model() as model:
        # Convert time_idx to standardized time for numerical stability
        time_std = (np.array(time_idx) - np.mean(time_idx)) / np.std(time_idx)
        
        # Much simpler model structure
        
        # Factor drift (improvement rate)
        mu = pm.HalfNormal('mu', sigma=0.1, shape=(n_factors,))
        
        # Factor noise
        sigma_f = pm.HalfNormal('sigma_f', sigma=0.2, shape=(n_factors,))
        
        # Factor loadings (all positive for simplicity)
        Lambda = pm.HalfNormal('Lambda', sigma=1.0, shape=(n_tasks, n_factors))
        
        # Factor values at each observation (not each time point)
        # This avoids complex tensor indexing
        f_at_obs = pm.Normal('f_at_obs', 
                            mu=-2.0 + mu[None, :] * time_std[:, None],
                            sigma=sigma_f[None, :],
                            shape=(len(y_obs), n_factors))
        
        # Task capabilities at each observation
        c = pt.sum(Lambda[task_idx, :] * f_at_obs, axis=1)
        
        # Observation noise
        tau = pm.HalfNormal('tau', sigma=0.3, shape=(n_tasks,))
        
        # Convert to logit space
        y_logit_obs = logit(y_obs)
        
        # Likelihood with explicit log_likelihood for WAIC
        y_logit = pm.Normal('y_logit', 
                           mu=c, 
                           sigma=tau[task_idx],
                           observed=y_logit_obs)
        
        # Explicit log-likelihood for WAIC
        ll = pm.Normal.logp(y_logit_obs, c, tau[task_idx])
        pm.Deterministic('log_likelihood', ll)
        
        # Store useful quantities
        pm.Deterministic('capabilities', c)
        pm.Deterministic('loadings', Lambda)
        pm.Deterministic('factors_at_obs', f_at_obs)
        
        # Reconstruct factor trajectories for plotting
        unique_times = np.unique(time_idx)
        f_traj = pt.zeros((n_factors, len(unique_times)))
        for i, t in enumerate(unique_times):
            obs_at_t = np.where(time_idx == t)[0]
            if len(obs_at_t) > 0:
                f_traj = pt.set_subtensor(f_traj[:, i], 
                                        pt.mean(f_at_obs[obs_at_t, :], axis=0))
        
        pm.Deterministic('factors', f_traj)
        
    return model

# 3) MODEL FITTING AND SELECTION ----------------------------
def fit_factor_models(y_obs, time_idx, task_idx, n_times, n_tasks, 
                     k_range=[2, 3, 4], n_samples=1000, n_tune=1500):
    """
    Fit models with different numbers of factors and compute WAIC
    """
    models = {}
    traces = {}
    waic_scores = {}
    
    for k in k_range:
        print(f"\nFitting model with {k} factors...")
        
        model = build_factor_model(y_obs, time_idx, task_idx, k, n_times, n_tasks)
        
        with model:
            # More robust sampling settings
            trace = pm.sample(n_samples, tune=n_tune, 
                            target_accept=0.95,
                            max_treedepth=12,
                            init="auto",
                            random_seed=42,
                            return_inferencedata=True)
            
            # Check sampling quality
            rhat = az.rhat(trace)
            max_rhat = float(rhat.max())
            n_divergences = trace.sample_stats.diverging.sum().item()
            
            print(f"  Max R-hat: {max_rhat:.3f}")
            print(f"  Divergences: {n_divergences}")
            
            # Compute WAIC (now with explicit log_likelihood)
            try:
                waic = az.waic(trace)
                waic_score = float(waic.waic)
                print(f"  WAIC: {waic_score:.2f}")
            except Exception as e:
                print(f"  WAIC computation failed: {e}")
                # Use LOO as fallback
                try:
                    loo = az.loo(trace)
                    waic_score = float(loo.loo)
                    print(f"  LOO (fallback): {waic_score:.2f}")
                except:
                    waic_score = np.inf
                    print(f"  Model evaluation failed, using inf")
            
            models[k] = model
            traces[k] = trace
            waic_scores[k] = waic_score
    
    # Select best model (skip failed models)
    valid_scores = {k: v for k, v in waic_scores.items() if v != np.inf}
    if valid_scores:
        best_k = min(valid_scores.keys(), key=lambda k: valid_scores[k])
        print(f"\nBest model: K={best_k} (WAIC={valid_scores[best_k]:.2f})")
    else:
        print(f"\nNo valid models found, using K={k_range[0]}")
        best_k = k_range[0]
    
    return models, traces, waic_scores, best_k

# 4) ANALYSIS AND INTERPRETATION -------------------------
def analyze_factors(trace, task_ids, task_to_idx, dates, best_k):
    """
    Analyze and interpret factor model results
    """
    # Extract posterior means
    factors = trace.posterior['factors'].mean(dim=['chain', 'draw']).values
    loadings = trace.posterior['loadings'].mean(dim=['chain', 'draw']).values
    mu_posterior = trace.posterior['mu'].mean(dim=['chain', 'draw']).values
    
    # Print factor interpretation
    print(f"\n=== FACTOR ANALYSIS (K={best_k}) ===")
    print("\nFactor drift rates (capability improvement):")
    for k in range(best_k):
        print(f"  Factor {k+1}: {mu_posterior[k]:.4f} per standardized time unit")
    
    print(f"\nFactor loadings (top 3 tasks per factor):")
    for k in range(best_k):
        loading_ranks = np.argsort(np.abs(loadings[:, k]))[::-1]
        print(f"\n  Factor {k+1}:")
        for i in range(min(3, len(task_ids))):
            task_idx_ranked = loading_ranks[i]
            if task_idx_ranked < len(task_ids):
                task_id = task_ids[task_idx_ranked]
                loading_val = loadings[task_idx_ranked, k]
                print(f"    Task {task_id}: {loading_val:.3f}")
    
    # Check sampling diagnostics
    rhat = az.rhat(trace)
    max_rhat = float(rhat.max()) if hasattr(rhat, 'max') else 1.0
    n_divergences = int(trace.sample_stats.diverging.sum()) if hasattr(trace.sample_stats, 'diverging') else 0
    
    print(f"\n=== SAMPLING DIAGNOSTICS ===")
    print(f"Max R-hat: {max_rhat:.3f}")
    print(f"Divergences: {n_divergences}")
    
    return factors, loadings, mu_posterior

# 5) PLOTTING UTILITIES --------------------------------------
def plot_raw_data_multi(time_points, y_obs, time_idx, task_idx, task_ids, dates):
    plt.figure(figsize=(12,8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(task_ids)))
    
    for i, tid in enumerate(task_ids):
        idx = np.where(task_idx == i)[0]
        if len(idx) > 0:
            plt.scatter(time_points[time_idx[idx]], y_obs[idx],
                       color=colors[i], s=60, alpha=0.7, label=f'Task {tid}')
    
    plt.xticks(time_points, [d.strftime('%Y-%m') for d in dates],
               rotation=45, ha='right')
    plt.title('Raw Test Scores by Task')
    plt.xlabel('Publication Date')
    plt.ylabel('Score')
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_factor_evolution(factors, dates, mu_posterior, best_k):
    """
    Plot factor evolution over time
    """
    fig, axes = plt.subplots(1, best_k, figsize=(5*best_k, 4))
    if best_k == 1:
        axes = [axes]
    
    for k in range(best_k):
        axes[k].plot(dates, factors[k, :], 'o-', color=COLOR_PALETTE[k % len(COLOR_PALETTE)])
        axes[k].set_title(f'Factor {k+1}\n(drift: {mu_posterior[k]:.4f}/day)')
        axes[k].set_xlabel('Date')
        axes[k].set_ylabel('Factor Level')
        axes[k].grid(alpha=0.3)
        axes[k].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_factor_loadings(loadings, task_ids, best_k):
    """
    Plot factor loading matrix as heatmap
    """
    plt.figure(figsize=(max(6, best_k*2), max(6, len(task_ids)*0.3)))
    im = plt.imshow(loadings, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im, label='Loading')
    plt.yticks(range(len(task_ids)), [f'Task {tid}' for tid in task_ids])
    plt.xticks(range(best_k), [f'Factor {k+1}' for k in range(best_k)])
    plt.title('Factor Loading Matrix')
    plt.tight_layout()
    plt.show()

# 6) TASK CLUSTERING BASED ON LOADINGS -------------------
def cluster_tasks_by_loadings(loadings, task_ids, n_clusters=3):
    """
    Cluster tasks based on factor loading patterns
    """
    from sklearn.cluster import KMeans
    
    # Cluster tasks based on loadings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(loadings)
    
    print(f"\n=== TASK CLUSTERING ===")
    for c in range(n_clusters):
        task_indices = np.where(clusters == c)[0]
        cluster_tasks = [task_ids[i] for i in task_indices]
        print(f"\nCluster {c+1}: {cluster_tasks}")
    
    return clusters

# 7) MAIN EXECUTION PIPELINE ------------------------------
def run_factor_analysis(df, task_ids, k_range=[2, 3], start_simple=True):
    """
    Complete pipeline for factor analysis
    """
    print("=== BAYESIAN DYNAMIC FACTOR MODEL ANALYSIS ===\n")
    
    # 1. Prepare data
    print("1. Preparing data...")
    time_points, y_obs, time_idx, task_idx, dates, task_to_idx = prepare_data_multi(df, task_ids)
    n_times = len(time_points)
    n_tasks = len(task_ids)
    n_obs = len(y_obs)
    
    print(f"   Data: {n_obs} observations, {n_tasks} tasks, {n_times} time points")
    
    # Check data coverage
    print(f"   Score range: [{y_obs.min():.3f}, {y_obs.max():.3f}]")
    print(f"   Tasks with data: {len(np.unique(task_idx))}/{n_tasks}")
    
    # 2. Plot raw data
    print("\n2. Plotting raw data...")
    plot_raw_data_multi(time_points, y_obs, time_idx, task_idx, task_ids, dates)
    
    # 3. Fit models (start simple if requested)
    if start_simple and len(k_range) > 1:
        print(f"\n3. Fitting factor models (starting with K={k_range[0]})...")
        # Try simplest model first
        try:
            models, traces, waic_scores, best_k = fit_factor_models(
                y_obs, time_idx, task_idx, n_times, n_tasks, k_range=[k_range[0]])
            
            # If successful, try other values
            if traces[k_range[0]] is not None:
                print("First model successful, trying other K values...")
                models_full, traces_full, waic_scores_full, best_k_full = fit_factor_models(
                    y_obs, time_idx, task_idx, n_times, n_tasks, k_range)
                models.update(models_full)
                traces.update(traces_full) 
                waic_scores.update(waic_scores_full)
                best_k = best_k_full
                
        except Exception as e:
            print(f"Error in model fitting: {e}")
            print("Trying with reduced complexity...")
            models, traces, waic_scores, best_k = fit_factor_models(
                y_obs, time_idx, task_idx, n_times, n_tasks, [2])
    else:
        print("\n3. Fitting factor models...")
        models, traces, waic_scores, best_k = fit_factor_models(
            y_obs, time_idx, task_idx, n_times, n_tasks, k_range)
    
    # 4. Analyze best model
    print("\n4. Analyzing best model...")
    factors, loadings, mu_posterior = analyze_factors(
        traces[best_k], task_ids, task_to_idx, dates, best_k)
    
    # 5. Plot results
    print("\n5. Plotting results...")
    plot_factor_evolution(factors, dates, mu_posterior, best_k)
    plot_factor_loadings(loadings, task_ids, best_k)
    
    # 6. Cluster tasks
    print("\n6. Clustering tasks...")
    n_clusters = min(3, len(task_ids)//2)  # Reasonable number of clusters
    clusters = cluster_tasks_by_loadings(loadings, task_ids, n_clusters)
    
    return {
        'models': models,
        'traces': traces, 
        'waic_scores': waic_scores,
        'best_k': best_k,
        'factors': factors,
        'loadings': loadings,
        'clusters': clusters,
        'data': (time_points, y_obs, time_idx, task_idx, dates, task_to_idx)
    }

# Example usage:

# Load your data
df = pd.read_csv('../../results/tables/df_model_test_scores.csv')

# Select subset of tasks for analysis (start small)
available_tasks = df['task_id'].unique()

# Start with fewer tasks and simpler models for testing
task_ids = available_tasks[:100]  # Start with 5 tasks

print(f"Selected tasks: {task_ids}")
print(f"Available data points per task:")
for tid in task_ids:
    n_points = len(df[df['task_id'] == tid].dropna(subset=['score']))
    print(f"  Task {tid}: {n_points} observations")

# Run complete analysis with conservative settings
results = run_factor_analysis(df, task_ids, k_range=[2, 3])

print("\n=== ANALYSIS COMPLETE ===")
print(f"Best model has {results['best_k']} factors")
print("Check plots and clustering results above!")

# Display WAIC comparison
print(f"\nModel comparison (WAIC):")
for k, waic in results['waic_scores'].items():
    marker = " <-- BEST" if k == results['best_k'] else ""
    print(f"  K={k}: {waic:.2f}{marker}")


############ PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv('../../results/tables/df_model_test_scores.csv')
df['Publication date'] = pd.to_datetime(df['Publication date'])
df = df.sort_values('Publication date')

# df = df[df['model'] != 'claude-3-5-sonnet-202410']

# Create task x time matrix
print("Creating task x time matrix...")
task_time_matrix = df.pivot_table(
    values='score', 
    index='task_id', 
    columns='Publication date', 
    aggfunc='mean'  # In case of duplicates
)

print(f"Matrix shape: {task_time_matrix.shape} (tasks x time)")
print(f"Missing values: {task_time_matrix.isna().sum().sum()}")

# Handle missing values - replace with zero
task_time_matrix = task_time_matrix.fillna(0)
print(f"After filling NaN with 0: {task_time_matrix.shape}")

# Use the cleaned matrix
X = task_time_matrix.values  # Tasks x Time
task_ids = task_time_matrix.index.values
time_points = task_time_matrix.columns

print(f"Final matrix: {X.shape} tasks x time points")

# Plot average performance across all tasks over time
plt.figure(figsize=(12, 6))

# Calculate mean and std across tasks for each time point (keeping zeros as zeros)
mean_scores = task_time_matrix.mean(axis=0)
std_scores = task_time_matrix.std(axis=0)
n_tasks_per_time = [len(task_ids)] * len(time_points)

# Plot mean with error bars
plt.errorbar(range(len(time_points)), mean_scores, yerr=std_scores, 
             fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8,
             color='steelblue', label='Mean ± Std')

# Add individual task trajectories (very faint)
for i in range(min(20, len(task_ids))):  # Show max 20 tasks to avoid clutter
    task_series = task_time_matrix.iloc[i]
    plt.plot(range(len(time_points)), task_series, 
            alpha=0.1, color='gray', linewidth=0.5)

plt.xlabel('Time Point')
plt.ylabel('Score')
plt.title('Average Performance Across All Tasks Over Time')
plt.xticks(range(len(time_points)), 
           [d.strftime('%Y-%m') for d in time_points], 
           rotation=45, ha='right')
plt.grid(alpha=0.3)
plt.legend()

# Add text showing number of tasks per time point
for i, n in enumerate(n_tasks_per_time):
    plt.text(i, plt.ylim()[0] + 0.02*(plt.ylim()[1]-plt.ylim()[0]), 
             f'n={n}', ha='center', va='bottom', fontsize=8, alpha=0.7)

plt.tight_layout()
plt.show()

print(f"Average performance over time (including zeros):")
for i, (date, score) in enumerate(zip(time_points, mean_scores)):
    print(f"{date.strftime('%Y-%m')}: {score:.1f} (all {len(task_ids)} tasks)")
print()

# Standardize the time series (across time for each task)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.T).T  # Standardize each task's time series

# PCA Analysis
print("\nRunning PCA...")
pca = PCA()
pca_scores = pca.fit_transform(X_scaled)  # Tasks in PC space

# Explained variance
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print(f"PC1 explains {explained_var[0]:.1%} of variance")
print(f"PC2 explains {explained_var[1]:.1%} of variance") 
print(f"PC3 explains {explained_var[2]:.1%} of variance")
print(f"First 3 PCs explain {cumulative_var[2]:.1%} of variance")

# Plot explained variance
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(explained_var)+1), explained_var, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_var)+1), cumulative_var, 'ro-')
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Variance Explained')
plt.axhline(y=0.8, color='k', linestyle='--', alpha=0.5, label='80%')
plt.axhline(y=0.9, color='k', linestyle='--', alpha=0.5, label='90%')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Clustering based on first few PCs
n_components = 3  # Use first 3 PCs
pca_subset = pca_scores[:, :n_components]

# Try different numbers of clusters
max_clusters = min(10, len(task_ids)//2)
inertias = []
silhouette_scores = []

from sklearn.metrics import silhouette_score

for k in range(2, max_clusters+1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(pca_subset)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(pca_subset, cluster_labels)
    silhouette_scores.append(sil_score)

# Plot elbow curve
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(2, max_clusters+1), inertias, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(2, max_clusters+1), silhouette_scores, 'go-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Choose optimal number of clusters (highest silhouette score)
optimal_k = range(2, max_clusters+1)[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters: {optimal_k}")

# Final clustering
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans_final.fit_predict(pca_subset)

# Visualize clusters in PC space
plt.figure(figsize=(15, 5))

# PC1 vs PC2
plt.subplot(1, 3, 1)
scatter = plt.scatter(pca_scores[:, 0], pca_scores[:, 1], c=cluster_labels, cmap='tab10', s=60, alpha=0.7)
plt.xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
plt.ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
plt.title('Tasks in PC1-PC2 Space')
plt.colorbar(scatter, label='Cluster')
plt.grid(alpha=0.3)

# PC1 vs PC3  
plt.subplot(1, 3, 2)
scatter = plt.scatter(pca_scores[:, 0], pca_scores[:, 2], c=cluster_labels, cmap='tab10', s=60, alpha=0.7)
plt.xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
plt.ylabel(f'PC3 ({explained_var[2]:.1%} variance)')
plt.title('Tasks in PC1-PC3 Space')
plt.colorbar(scatter, label='Cluster')
plt.grid(alpha=0.3)

# PC2 vs PC3
plt.subplot(1, 3, 3)
scatter = plt.scatter(pca_scores[:, 1], pca_scores[:, 2], c=cluster_labels, cmap='tab10', s=60, alpha=0.7)
plt.xlabel(f'PC2 ({explained_var[1]:.1%} variance)')
plt.ylabel(f'PC3 ({explained_var[2]:.1%} variance)')
plt.title('Tasks in PC2-PC3 Space')
plt.colorbar(scatter, label='Cluster')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Show cluster assignments
print(f"\nCluster assignments:")
cluster_df = pd.DataFrame({
    'task_id': task_ids,
    'cluster': cluster_labels,
    'PC1': pca_scores[:, 0],
    'PC2': pca_scores[:, 1], 
    'PC3': pca_scores[:, 2]
})

for cluster_id in range(optimal_k):
    tasks_in_cluster = cluster_df[cluster_df['cluster'] == cluster_id]['task_id'].values
    print(f"Cluster {cluster_id}: {len(tasks_in_cluster)} tasks")
    print(f"  Tasks: {tasks_in_cluster[:10]}{'...' if len(tasks_in_cluster) > 10 else ''}")

# Plot average time series by cluster
plt.figure(figsize=(12, 8))
colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))

for cluster_id in range(optimal_k):
    cluster_tasks = task_ids[cluster_labels == cluster_id]
    cluster_data = task_time_matrix.loc[cluster_tasks]
    
    # Plot individual tasks (thin lines)
    for _, task_series in cluster_data.iterrows():
        plt.plot(range(len(time_points)), task_series.values, 
                color=colors[cluster_id], alpha=0.2, linewidth=0.5)
    
    # Plot cluster average (thick line)  
    cluster_mean = cluster_data.mean(axis=0)
    plt.plot(range(len(time_points)), cluster_mean.values,
            color=colors[cluster_id], linewidth=3, 
            label=f'Cluster {cluster_id} (n={len(cluster_tasks)})')

plt.xlabel('Time Point')
plt.ylabel('Score')
plt.title('Task Clusters: Time Series Patterns')
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(range(len(time_points)), 
           [d.strftime('%Y-%m') for d in time_points], 
           rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Principal components interpretation
print(f"\nPrincipal Component Loadings (first 3 PCs):")
pc_loadings = pca.components_[:3, :].T  # Time x PC
loadings_df = pd.DataFrame(pc_loadings, 
                          index=[d.strftime('%Y-%m') for d in time_points],
                          columns=['PC1', 'PC2', 'PC3'])
print(loadings_df.round(3))

# Plot PC loadings
plt.figure(figsize=(12, 4))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.plot(range(len(time_points)), pc_loadings[:, i], 'o-', linewidth=2)
    plt.xlabel('Time Point')
    plt.ylabel(f'PC{i+1} Loading')
    plt.title(f'PC{i+1} ({explained_var[i]:.1%} variance)')
    plt.grid(alpha=0.3)
    plt.xticks(range(len(time_points)), 
               [d.strftime('%Y-%m') for d in time_points], 
               rotation=45, ha='right')

plt.tight_layout()
plt.show()

print(f"\nSummary:")
print(f"- Analyzed {len(task_ids)} tasks across {len(time_points)} time points")
print(f"- First 3 PCs explain {cumulative_var[2]:.1%} of variance")
print(f"- Optimal clustering: {optimal_k} clusters")
print(f"- Cluster assignments saved in cluster_df DataFrame")


# Show cluster assignments
print(f"\nCluster assignments:")
cluster_df = pd.DataFrame({
    'task_id': task_ids,
    'cluster': cluster_labels,
    'PC1': pca_scores[:, 0],
    'PC2': pca_scores[:, 1], 
    'PC3': pca_scores[:, 2]
})

for cluster_id in range(optimal_k):
    tasks_in_cluster = cluster_df[cluster_df['cluster'] == cluster_id]['task_id'].values
    print(f"Cluster {cluster_id}: {len(tasks_in_cluster)} tasks")
    print(f"  Tasks: {tasks_in_cluster[:10]}{'...' if len(tasks_in_cluster) > 10 else ''}")

# Create word clouds for each cluster
print(f"\nCreating word clouds for cluster task descriptions...")

# Get unique task descriptions (avoid duplicates from same task_id)
task_descriptions = df.groupby('task_id')['task_description'].first().reset_index()

# Merge with cluster assignments
cluster_descriptions = cluster_df.merge(task_descriptions, on='task_id', how='left')

# Check for missing descriptions
missing_desc = cluster_descriptions['task_description'].isna().sum()
if missing_desc > 0:
    print(f"Warning: {missing_desc} tasks missing descriptions")
    cluster_descriptions = cluster_descriptions.dropna(subset=['task_description'])

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Create subplots for word clouds
fig, axes = plt.subplots(2, (optimal_k + 1) // 2, figsize=(15, 8))
if optimal_k == 1:
    axes = [axes]
elif optimal_k <= 2:
    axes = axes.flatten()
else:
    axes = axes.flatten()

for cluster_id in range(optimal_k):
    # Get descriptions for this cluster
    cluster_desc = cluster_descriptions[cluster_descriptions['cluster'] == cluster_id]
    
    if len(cluster_desc) == 0:
        continue
        
    # Combine all descriptions into one text
    combined_text = ' '.join(cluster_desc['task_description'].fillna(''))
    
    if len(combined_text.strip()) == 0:
        continue
    
    # Create word cloud
    wordcloud = WordCloud(
        width=400, height=300,
        background_color='white',
        max_words=50,
        colormap='viridis',
        stopwords={'and', 'or', 'the', 'to', 'of', 'for', 'in', 'on', 'with', 'by', 'from', 'such', 'as', 'that', 'this', 'is', 'are', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'other', 'others', 'all', 'any', 'some', 'each', 'every', 'their', 'them', 'they', 'its', 'it', 'his', 'her', 'him', 'she', 'he', 'we', 'us', 'our', 'you', 'your'}
    ).generate(combined_text)
    
    # Plot
    ax = axes[cluster_id] if optimal_k > 1 else axes[0]
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(f'Cluster {cluster_id}\n({len(cluster_desc)} tasks)', fontsize=12)
    ax.axis('off')

# Hide unused subplots
for i in range(optimal_k, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# Print detailed cluster descriptions
print(f"\nDetailed cluster analysis:")
for cluster_id in range(optimal_k):
    cluster_tasks = cluster_descriptions[cluster_descriptions['cluster'] == cluster_id]
    print(f"\n{'='*50}")
    print(f"CLUSTER {cluster_id} ({len(cluster_tasks)} tasks)")
    print(f"{'='*50}")
    
    # Show sample descriptions
    print("Sample task descriptions:")
    for i, (_, row) in enumerate(cluster_tasks.head(5).iterrows()):
        print(f"  {i+1}. Task {row['task_id']}: {row['task_description']}")
    
    if len(cluster_tasks) > 5:
        print(f"  ... and {len(cluster_tasks)-5} more tasks")
    
    # Also show occupation groups if available
    if 'occupation_group' in df.columns:
        cluster_full = cluster_df.merge(df[['task_id', 'occupation_group']].drop_duplicates(), on='task_id', how='left')
        cluster_occs = cluster_full[cluster_full['cluster'] == cluster_id]['occupation_group'].value_counts()
        print(f"\nOccupation groups in this cluster:")
        for occ, count in cluster_occs.head(5).items():
            print(f"  - {occ}: {count} tasks")


# Plot average time series by cluster
plt.figure(figsize=(12, 8))
colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))

for cluster_id in range(optimal_k):
    cluster_tasks = task_ids[cluster_labels == cluster_id]
    cluster_data = task_time_matrix.loc[cluster_tasks]
    
    # Plot individual tasks (thin lines)
    for _, task_series in cluster_data.iterrows():
        plt.plot(range(len(time_points)), task_series.values, 
                color=colors[cluster_id], alpha=0.2, linewidth=0.5)
    
    # Plot cluster average (thick line)  
    cluster_mean = cluster_data.mean(axis=0)
    plt.plot(range(len(time_points)), cluster_mean.values,
            color=colors[cluster_id], linewidth=3, 
            label=f'Cluster {cluster_id} (n={len(cluster_tasks)})')

plt.xlabel('Time Point')
plt.ylabel('Score')
plt.title('Task Clusters: Time Series Patterns')
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(range(len(time_points)), 
           [d.strftime('%Y-%m') for d in time_points], 
           rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Principal components interpretation
print(f"\nPrincipal Component Loadings (first 3 PCs):")
pc_loadings = pca.components_[:3, :].T  # Time x PC
loadings_df = pd.DataFrame(pc_loadings, 
                          index=[d.strftime('%Y-%m') for d in time_points],
                          columns=['PC1', 'PC2', 'PC3'])
print(loadings_df.round(3))

# Plot PC loadings
plt.figure(figsize=(12, 4))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.plot(range(len(time_points)), pc_loadings[:, i], 'o-', linewidth=2)
    plt.xlabel('Time Point')
    plt.ylabel(f'PC{i+1} Loading')
    plt.title(f'PC{i+1} ({explained_var[i]:.1%} variance)')
    plt.grid(alpha=0.3)
    plt.xticks(range(len(time_points)), 
               [d.strftime('%Y-%m') for d in time_points], 
               rotation=45, ha='right')

plt.tight_layout()
plt.show()

print(f"\nSummary:")
print(f"- Analyzed {len(task_ids)} tasks across {len(time_points)} time points")
print(f"- First 3 PCs explain {cumulative_var[2]:.1%} of variance")
print(f"- Optimal clustering: {optimal_k} clusters")
print(f"- Cluster assignments saved in cluster_df DataFrame")

unique_combinations = df[['Publication date', 'model']].drop_duplicates()
print(unique_combinations)

df['score'][df['model'] == 'gpt-3.5-turbo-0125'].hist(bins=20, density=True)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Extract the scores for the given model and drop missing values
scores = df.loc[df['model'] == 'gpt-3.5-turbo-0125', 'score'].dropna()

# Fit a Gaussian distribution: returns (mu, sigma)
mu, sigma = norm.fit(scores)
print("Fitted Gaussian parameters:")
print("mu =", mu)
print("sigma =", sigma)
