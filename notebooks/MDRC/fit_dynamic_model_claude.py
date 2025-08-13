'''
SO far this is the best apprich better that fit_dyanmic_model.py
'''
#%%
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from typing import Optional, List, Union
import warnings
warnings.filterwarnings('ignore')

def prepare_data_multi(df, task_ids, handle_nan='ignore', fill_pct=0.01):
    """Prepare data for multi-task modeling"""
    df2 = df.loc[df['task_id'].isin(task_ids),
                 ['task_id', 'Publication date', 'score']].copy()
    df2['Publication date'] = pd.to_datetime(df2['Publication date'])
    df2.sort_values('Publication date', inplace=True)

    dates = df2['Publication date'].drop_duplicates().sort_values()
    pivot = (df2.pivot(index='Publication date', columns='task_id', values='score')
                 .reindex(dates)
                 .reindex(columns=task_ids))

    if handle_nan == 'fill':
        lo, hi = 100*fill_pct, 100*(1 - fill_pct)
        pivot = pivot.fillna(lo).replace({0: lo, 100: hi})
    elif handle_nan == 'zero':
        pivot = pivot.fillna(0)
    else:                 # 'ignore'
        pivot = pivot.dropna()

    first = dates.iloc[0]
    time_points = (dates - first).dt.days.to_numpy()
    y_obs = pivot.values / 100.0
    
    # Replace exactly 0 and 1 with 0.01 and 0.99 to avoid sigmoid issues
    y_obs = np.where(y_obs == 0.0, 0.01, y_obs)
    y_obs = np.where(y_obs == 1.0, 0.99, y_obs)

    return time_points, y_obs, list(dates)


def build_factor_model(tp, y_obs, task_ids, k=3, grounding_tasks=None, 
                      model_name="LLM_Factor_Model",
                      # Prior hyperparameters
                      mu_prior=(0.04, 0.04),      # (mean, std) for TruncatedNormal
                      sigma_prior=0.5,            # beta for HalfCauchy  
                      tau_prior=0.05,             # beta for HalfCauchy
                      f0_prior=None,         # (mean, std) for initial factors
                      g_sigmoid=0.0919,           # sigmoid steepness parameter
                      dirichlet_alpha=0.5,        # Dirichlet concentration parameter
                      use_halfnormal=False,       # Use HalfNormal instead of Dirichlet
                      halfnormal_sigma=1.0):      # HalfNormal scale parameter
    """
    Build dynamic factor model for LLM capabilities
    
    Parameters:
    -----------
    tp : array-like
        Time points (days from first observation)
    y_obs : array-like 
        Observed scores (T x N_tasks)
    task_ids : list
        List of task IDs corresponding to columns in y_obs
    k : int
        Number of factors (2 or 3)
    grounding_tasks : list, optional
        Task IDs for identification constraints. If None, uses defaults.
    model_name : str
        Name for the PyMC model
        
    Prior Parameters:
    ----------------
    mu_prior : tuple
        (mean, std) for drift parameter TruncatedNormal prior
    sigma_prior : float  
        beta parameter for innovation variance HalfCauchy prior
    tau_prior : float
        beta parameter for observation error HalfCauchy prior
    f0_prior : tuple
        (mean, std) for initial factor values Normal prior
    g_sigmoid : float
        Sigmoid transformation steepness parameter
    dirichlet_alpha : float
        Concentration parameter for Dirichlet factor loading priors
    use_halfnormal : bool
        If True, use independent HalfNormal priors instead of Dirichlet
    halfnormal_sigma : float
        Scale parameter for HalfNormal factor loading priors
        
    Returns:
    --------
    model : PyMC model
    """
    
    print(f"\n=== Building Factor Model ===")
    print(f"k={k} factors, {len(task_ids)} tasks, {len(tp)} time points")
    
    # Set default grounding tasks
    if grounding_tasks is None:
        default_grounding = {
            2: [18868, 15735],
            3: [18868, 15735, 1094]
        }
        grounding_tasks = default_grounding[k]
    
    if len(grounding_tasks) != k:
        raise ValueError(f"grounding_tasks must have length k={k}")
    
    print(f"Grounding tasks: {grounding_tasks}")
    print(f"Task IDs type: {type(task_ids[0])}, first few: {task_ids[:5]}")
    
    # Convert grounding tasks to same type as task_ids
    if len(task_ids) > 0:
        task_id_type = type(task_ids[0])
        grounding_tasks = [task_id_type(t) for t in grounding_tasks]
        print(f"Converted grounding tasks: {grounding_tasks}")
    
    # Get dimensions
    T, N = y_obs.shape
    print(f"Data dimensions: T={T}, N={N}")
    
    # Create mapping from task_id to index
    task_to_idx = {task_id: i for i, task_id in enumerate(task_ids)}
    grounding_indices = [task_to_idx[task_id] for task_id in grounding_tasks]
    
    # Check all grounding tasks are in our data
    missing_tasks = [t for t in grounding_tasks if t not in task_to_idx]
    if missing_tasks:
        print(f"Available task_ids: {sorted(task_ids)[:10]}...")
        raise ValueError(f"Grounding tasks {missing_tasks} not found in task_ids")
    
    print(f"Grounding task indices: {grounding_indices}")
    
    # Compute time differences for random walk
    time_diffs = np.diff(np.concatenate([[0], tp]))  # [Δt₁, Δt₂, ..., ΔtT]
    print(f"Time differences: {time_diffs}")
    
    with pm.Model(name=model_name) as model:
        
        print("Setting up priors...")
        print(f"  mu_prior: N({mu_prior[0]}, {mu_prior[1]}) truncated at 0")
        print(f"  sigma_prior: HalfCauchy({sigma_prior})")
        print(f"  tau_prior: HalfCauchy({tau_prior})")
        # print(f"  f0_prior: N({f0_prior[0]}, {f0_prior[1]})")
        print(f"  g_sigmoid: {g_sigmoid}")
        
        if use_halfnormal:
            print(f"  factor loadings: HalfNormal(sigma={halfnormal_sigma})")
        else:
            print(f"  factor loadings: Dirichlet(alpha={dirichlet_alpha})")
        
        # Prior hyperparameters using configurable values
        mu = pm.TruncatedNormal('mu', mu=mu_prior[0], sigma=mu_prior[1], lower=0, 
                               shape=k, initval=np.full(k, mu_prior[0]))
        sigma = pm.HalfCauchy('sigma', beta=sigma_prior, shape=k)
        tau = pm.HalfCauchy('tau', beta=tau_prior)

        # Initial factor values  
        if f0_prior is None:
            # Default factor-specific priors based on typical performance levels
            if k == 2:
                f0_means = [15, -25]
                f0_sds = [10, 10]
            elif k == 3:
                f0_means = [0, -15, -50]
                f0_sds = [2, 5, 10]
            else:
                f0_means = [0] * k
                f0_sds = [5] * k
        else:
            # User provided list of (mean, std) tuples
            f0_means = [prior[0] for prior in f0_prior]
            f0_sds = [prior[1] for prior in f0_prior]

        f0 = pm.Normal('f0', mu=f0_means, sigma=f0_sds, shape=k, 
                    initval=np.array(f0_means))
        print(f"  f0_prior: means={f0_means}, sds={f0_sds}")
        
        # # Initial factor values  
        # f0 = pm.Normal('f0', mu=f0_prior[0], sigma=f0_prior[1], shape=k, 
        #               initval=np.full(k, f0_prior[0]))
        
        print("Setting up factor evolution...")
        
        # Factor evolution: implement random walk with uneven time steps
        # f[0] ~ N(f0, small_variance) and f[t] = f[t-1] + N(Δt·μ, Δt·σ²)
        f_list = [f0]  # Start with initial values
        

        for t in range(1, T):
            dt = time_diffs[t]  # Time difference from previous observation
            f_prev = f_list[-1]
            
            # Non-centered: sample standard normal first, then transform
            eta = pm.Normal(f'eta_{t}', mu=0, sigma=1, shape=k)
            f_increment = dt * mu + pt.sqrt(dt) * sigma * eta
            f_current = f_prev + f_increment
            f_list.append(f_current)
        
        # Stack all factors into a matrix
        f_all = pt.stack(f_list, axis=0)  # Shape: (T, k)
        
        print("Setting up factor loadings...")
        
        if use_halfnormal:
            # Vectorized HalfNormal loadings: single (N x k) tensor
            print(f"  Using vectorized HalfNormal loadings (shape: {N}x{k})")
            raw_loadings = pm.HalfNormal('loadings_raw', sigma=halfnormal_sigma, 
                                        shape=(N, k))
            loadings = raw_loadings  # Start with raw loadings
            
        else:
            # Vectorized Dirichlet loadings: single (N x k) tensor  
            print(f"  Using vectorized Dirichlet loadings (shape: {N}x{k})")
            raw_loadings = pm.Dirichlet('loadings_raw', 
                                       a=np.full((N, k), dirichlet_alpha),
                                       shape=(N, k))
            loadings = raw_loadings  # Start with raw loadings
        
        # Apply identification constraints using set_subtensor
        print("  Applying identification constraints...")
        
        # First grounding task: loads only on factor 1
        # Set to [1, 0, 0] for k=3 or [1, 0] for k=2
        grounding_idx_0 = grounding_indices[0]
        if k == 2:
            constraint_0 = pt.as_tensor([1.0, 0.0])
        else:  # k == 3
            constraint_0 = pt.as_tensor([1.0, 0.0, 0.0])
        loadings = pt.set_subtensor(loadings[grounding_idx_0], constraint_0)
        print(f"    Task {grounding_tasks[0]} (idx {grounding_idx_0}): [1, 0, ...]")
        
        # Second grounding task: loads on factors 1,2 only
        grounding_idx_1 = grounding_indices[1]
        if use_halfnormal:
            # Keep the first two loadings, set third to zero (if k=3)
            if k == 3:
                loadings = pt.set_subtensor(loadings[grounding_idx_1, 2], 0.0)
        else:
            # For Dirichlet: normalize first two components, set third to zero
            if k == 2:
                # Normalize the two components to sum to 1
                raw_12 = loadings[grounding_idx_1, :2]
                normalized_12 = raw_12 / pt.sum(raw_12)
                loadings = pt.set_subtensor(loadings[grounding_idx_1, :2], normalized_12)
            else:  # k == 3
                # Normalize first two components, set third to zero
                raw_12 = loadings[grounding_idx_1, :2] 
                normalized_12 = raw_12 / pt.sum(raw_12)
                loadings = pt.set_subtensor(loadings[grounding_idx_1, :2], normalized_12)
                loadings = pt.set_subtensor(loadings[grounding_idx_1, 2], 0.0)
        print(f"    Task {grounding_tasks[1]} (idx {grounding_idx_1}): [λ₁, λ₂, 0]")
        
        # Third grounding task (only for k=3): loads on all factors
        if k == 3:
            grounding_idx_2 = grounding_indices[2]
            if not use_halfnormal:
                # For Dirichlet: ensure it sums to 1 (should already, but make explicit)
                raw_123 = loadings[grounding_idx_2, :]
                normalized_123 = raw_123 / pt.sum(raw_123)
                loadings = pt.set_subtensor(loadings[grounding_idx_2, :], normalized_123)
            # For HalfNormal: keep as is
            print(f"    Task {grounding_tasks[2]} (idx {grounding_idx_2}): [λ₁, λ₂, λ₃]")
        
        print("Setting up observation model...")
        
        # Store final loadings matrix
        pm.Deterministic('loadings_matrix', loadings)
        
        # Underlying capabilities: c_it = Σ λ_ik * f_kt
        capabilities = pt.dot(loadings, f_all.T).T  # Shape: (T, N)
        
        # Expected scores through sigmoid transformation
        mu_scores = 1 / (1 + pt.exp(-g_sigmoid * capabilities))
        
        # Observed scores with measurement error
        y_hat = pm.Normal('y_obs', mu=mu_scores, sigma=tau, 
                         observed=y_obs, shape=(T, N))
        
        # Store useful quantities
        pm.Deterministic('capabilities', capabilities)
        pm.Deterministic('factors', f_all)
        
        print(f"Model built successfully! Variables: {len(model.named_vars)}")
        
    return model


def test_model_build(df, k=3, n_tasks=10):
    """Test model building with a subset of tasks"""
    
    # Select first n_tasks for testing
    tasks = df['task_id'].unique().tolist()[:n_tasks]
    
    # Prepare data with 'fill' option
    tp, y_obs, dates = prepare_data_multi(df, tasks, handle_nan='fill')
    
    print(f"Testing model with k={k}, {len(tasks)} tasks")
    print(f"Time points shape: {tp.shape}")
    print(f"Observations shape: {y_obs.shape}")
    print(f"Time points: {tp}")
    
    # Build model
    model = build_factor_model(tp, y_obs, tasks, k=k)
    
    print(f"\nModel built successfully!")
    print(f"Model variables: {list(model.named_vars.keys())}")
    
    return model, tp, y_obs, tasks


# Quick test function for the user to run
def quick_test_with_data(df, k=3, max_tasks=None, **prior_kwargs):
    """
    Quick test of the model with real data
    
    Parameters:
    -----------
    df : DataFrame
        Your dataframe with columns: task_id, score, Publication date
    k : int
        Number of factors (2 or 3)  
    max_tasks : int or None
        Maximum number of tasks to include (for faster testing). None = use all tasks
    **prior_kwargs : dict
        Additional prior parameters to pass to build_factor_model()
        e.g., mu_prior=(0.05, 0.03), sigma_prior=0.2, etc.
    """
    
    print("="*60)
    print(f"QUICK TEST: k={k} factor model")
    print("="*60)
    
    # Get tasks 
    all_tasks = df['task_id'].unique().tolist()
    if max_tasks and len(all_tasks) > max_tasks:
        tasks = all_tasks[:max_tasks]
        print(f"Using first {max_tasks} tasks out of {len(all_tasks)} available")
        
        # Check if grounding tasks are included
        grounding_defaults = {2: [18868, 1094], 3: [18868, 1094, 15735]}
        required_tasks = grounding_defaults[k]
        
        # Convert to same type as tasks
        if len(tasks) > 0:
            task_type = type(tasks[0])
            required_tasks = [task_type(t) for t in required_tasks]
        
        missing_grounding = [t for t in required_tasks if t not in tasks]
        if missing_grounding:
            print(f"⚠️  Adding missing grounding tasks: {missing_grounding}")
            tasks.extend(missing_grounding)
            tasks = list(set(tasks))  # Remove duplicates
            
    else:
        tasks = all_tasks
    
    print(f"Tasks to include: {len(tasks)}")
    
    # Prepare data
    print("Preparing data...")
    tp, y_obs, dates = prepare_data_multi(df, tasks, handle_nan='fill')
    
    print(f"Time points: {tp}")
    print(f"Data shape: {y_obs.shape}")
    print(f"Score ranges: [{y_obs.min():.3f}, {y_obs.max():.3f}]")
    
    # Build model
    print("Building model...")
    try:
        model = build_factor_model(tp, y_obs, tasks, k=k, **prior_kwargs)
        print("✅ Model built successfully!")
        return model, tp, y_obs, tasks
        
    except Exception as e:
        print(f"❌ Model building failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


#%%
df = pd.read_csv('../../results/tables/df_model_test_scores.csv')

# df = df[df['model'] != 'claude-3-5-sonnet-202410']

# Test with HalfNormal loadings (much simpler computational graph)
# Test the vectorized version with HalfNormal
model_vec, tp_vec, y_obs_vec, tasks_vec = quick_test_with_data(
    df, k=2, max_tasks=None,
    use_halfnormal=True,
    halfnormal_sigma=1.0
)

print("Trying sampling with vectorized HalfNormal loadings...")
with model_vec:
    trace_vec = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        target_accept=0.97,
        cores=4,
        return_inferencedata=True,
        random_seed=42
    )
#%%
import numpy as np
import pandas as pd
import arviz as az
import warnings
from collections import defaultdict

def analyze_fitted_parameters(trace, k=3):
    """
    Analyze only the fitted parameters (not deterministic quantities)
    
    Parameters:
    -----------
    trace : ArviZ InferenceData
        MCMC trace from sampling
    k : int
        Number of factors
    """
    
    # Suppress ArviZ warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = az.summary(trace)
    
    # Clean variable names
    def clean_name(var_name):
        if "::" in var_name:
            return var_name.split("::")[-1]
        return var_name
    
    print("="*80)
    print("FITTED PARAMETERS ANALYSIS")
    print("="*80)
    
    # 1. CORE MODEL PARAMETERS
    print("\n📊 CORE MODEL PARAMETERS")
    print("-"*60)
    
    # Drift parameters (mu)
    print("Drift Parameters (μ - daily improvement):")
    mu_vars = [v for v in summary.index if 'mu[' in v]
    for i in range(k):
        mu_var = None
        for v in mu_vars:
            if f'mu[{i}]' in v:
                mu_var = v
                break
        
        if mu_var:
            row = summary.loc[mu_var]
            status = "✅" if row['r_hat'] < 1.01 else "⚠️" if row['r_hat'] < 1.05 else "❌"
            print(f"  μ[{i}]: {row['mean']:>8.4f} ± {row['sd']:>7.4f}  "
                  f"R̂={row['r_hat']:>5.3f} {status}")
    
    # Innovation variances (sigma)
    print("\nInnovation Variances (σ - daily volatility):")
    sigma_vars = [v for v in summary.index if 'sigma[' in v]
    for i in range(k):
        sigma_var = None
        for v in sigma_vars:
            if f'sigma[{i}]' in v:
                sigma_var = v
                break
        
        if sigma_var:
            row = summary.loc[sigma_var]
            status = "✅" if row['r_hat'] < 1.01 else "⚠️" if row['r_hat'] < 1.05 else "❌"
            print(f"  σ[{i}]: {row['mean']:>8.4f} ± {row['sd']:>7.4f}  "
                  f"R̂={row['r_hat']:>5.3f} {status}")
    
    # Observation error (tau)
    print("\nObservation Error (τ):")
    tau_vars = [v for v in summary.index if v.endswith('tau')]
    if tau_vars:
        tau_row = summary.loc[tau_vars[0]]
        status = "✅" if tau_row['r_hat'] < 1.01 else "⚠️" if tau_row['r_hat'] < 1.05 else "❌"
        print(f"  τ:   {tau_row['mean']:>8.4f} ± {tau_row['sd']:>7.4f}  "
              f"R̂={tau_row['r_hat']:>5.3f} {status}")
    
    # Initial factor values (f0)
    print("\nInitial Factor Values (f₀):")
    f0_vars = [v for v in summary.index if 'f0[' in v]
    for i in range(k):
        f0_var = None
        for v in f0_vars:
            if f'f0[{i}]' in v:
                f0_var = v
                break
        
        if f0_var:
            row = summary.loc[f0_var]
            status = "✅" if row['r_hat'] < 1.01 else "⚠️" if row['r_hat'] < 1.05 else "❌"
            print(f"  f₀[{i}]: {row['mean']:>7.2f} ± {row['sd']:>7.2f}  "
                  f"R̂={row['r_hat']:>5.3f} {status}")
    
    # 2. FACTOR INCREMENTS
    print(f"\n⏱️  FACTOR INCREMENTS (Random Walk Steps)")
    print("-"*60)
    
    increment_vars = [v for v in summary.index if 'eta_' in v]
    
    # Parse variable names like "f_increment_1[0]" -> time=1, factor=0
    def parse_increment_var(var_name):
        # Extract time step and factor index from "f_increment_1[0]"
        parts = var_name.split('f_increment_')[-1]  # Get "1[0]"
        if '[' in parts:
            time_str = parts.split('[')[0]  # Get "1"
            factor_str = parts.split('[')[1].split(']')[0]  # Get "0"
            return int(time_str), int(factor_str)
        else:
            return int(parts), 0  # Fallback
    
    # Group by time step
    increments_by_time = defaultdict(list)
    
    for var in increment_vars:
        try:
            time_step, factor_idx = parse_increment_var(var)
            increments_by_time[time_step].append((var, factor_idx))
        except:
            # If parsing fails, just group by variable name
            increments_by_time['unknown'].append((var, 0))
    
    # Show first few time steps
    print("Sample of factor increments (first 3 time steps):")
    time_steps = sorted([t for t in increments_by_time.keys() if t != 'unknown'])
    
    for t in time_steps[:3]:
        print(f"  Time step {t}:")
        # Sort by factor index
        increments_by_time[t].sort(key=lambda x: x[1])
        
        for var, factor_idx in increments_by_time[t]:
            row = summary.loc[var]
            status = "✅" if row['r_hat'] < 1.01 else "⚠️" if row['r_hat'] < 1.05 else "❌"
            print(f"    Δf[{factor_idx}]: {row['mean']:>7.3f} ± {row['sd']:>6.3f}  "
                  f"R̂={row['r_hat']:>5.3f} {status}")
    
    if len(time_steps) > 3:
        print(f"  ... and {len(time_steps) - 3} more time steps")
    
    # Show increment convergence summary
    increment_summary = summary.loc[increment_vars]
    inc_good = (increment_summary['r_hat'] < 1.01).sum()
    inc_marginal = ((increment_summary['r_hat'] >= 1.01) & 
                   (increment_summary['r_hat'] < 1.05)).sum()
    inc_bad = (increment_summary['r_hat'] >= 1.05).sum()
    
    print(f"\nIncrement convergence summary:")
    print(f"  ✅ Good: {inc_good}/{len(increment_vars)} ({100*inc_good/len(increment_vars):.1f}%)")
    print(f"  ⚠️  Marginal: {inc_marginal}/{len(increment_vars)} ({100*inc_marginal/len(increment_vars):.1f}%)")
    print(f"  ❌ Poor: {inc_bad}/{len(increment_vars)} ({100*inc_bad/len(increment_vars):.1f}%)")
    
    # 3. FACTOR LOADINGS (RAW PARAMETERS)
    print(f"\n🔗 FACTOR LOADINGS (λ - Raw Parameters)")
    print("-"*60)
    
    loadings_vars = [v for v in summary.index if 'loadings_raw[' in v]
    print(f"Total loading parameters: {len(loadings_vars)}")
    
    # Get convergence stats for loadings
    loadings_summary = summary.loc[loadings_vars]
    rhat_good = (loadings_summary['r_hat'] < 1.01).sum()
    rhat_marginal = ((loadings_summary['r_hat'] >= 1.01) & 
                     (loadings_summary['r_hat'] < 1.05)).sum()
    rhat_bad = (loadings_summary['r_hat'] >= 1.05).sum()
    
    print(f"Loading convergence:")
    print(f"  ✅ Good (R̂ < 1.01):      {rhat_good:>4d}/{len(loadings_vars)} ({100*rhat_good/len(loadings_vars):.1f}%)")
    print(f"  ⚠️  Marginal (1.01 ≤ R̂ < 1.05): {rhat_marginal:>4d}/{len(loadings_vars)} ({100*rhat_marginal/len(loadings_vars):.1f}%)")
    print(f"  ❌ Poor (R̂ ≥ 1.05):     {rhat_bad:>4d}/{len(loadings_vars)} ({100*rhat_bad/len(loadings_vars):.1f}%)")
    
    print(f"\nLoading statistics:")
    print(f"  Mean loading value: {loadings_summary['mean'].mean():>6.3f}")
    print(f"  Loading range: [{loadings_summary['mean'].min():>6.3f}, {loadings_summary['mean'].max():>6.3f}]")
    print(f"  Max R̂: {loadings_summary['r_hat'].max():>6.3f}")
    print(f"  Min ESS: {loadings_summary['ess_bulk'].min():>6.0f}")
    
    # Show worst converging loadings
    worst_loadings = loadings_summary.nlargest(5, 'r_hat')
    print(f"\nWorst converging loadings:")
    for idx, (var_name, row) in enumerate(worst_loadings.iterrows()):
        clean_var = clean_name(var_name)
        status = "❌" if row['r_hat'] >= 1.05 else "⚠️"
        print(f"  {idx+1}. {clean_var:<25} {row['mean']:>7.3f} ± {row['sd']:>6.3f}  "
              f"R̂={row['r_hat']:>5.3f} {status}")
    
    # 4. OVERALL CONVERGENCE SUMMARY
    print(f"\n📈 OVERALL CONVERGENCE SUMMARY")
    print("-"*60)
    
    # Get only fitted parameters (exclude deterministics)
    fitted_vars = [v for v in summary.index if any(pattern in v for pattern in 
                   ['mu[', 'sigma[', 'tau', 'f0[', 'f_increment_', 'loadings_raw['])]
    
    fitted_summary = summary.loc[fitted_vars]
    
    total_fitted = len(fitted_vars)
    converged_fitted = (fitted_summary['r_hat'] < 1.01).sum()
    marginal_fitted = ((fitted_summary['r_hat'] >= 1.01) & 
                      (fitted_summary['r_hat'] < 1.05)).sum()
    poor_fitted = (fitted_summary['r_hat'] >= 1.05).sum()
    
    print(f"Fitted parameters: {total_fitted}")
    print(f"  ✅ Converged (R̂ < 1.01):    {converged_fitted:>4d} ({100*converged_fitted/total_fitted:.1f}%)")
    print(f"  ⚠️  Marginal (1.01 ≤ R̂ < 1.05): {marginal_fitted:>4d} ({100*marginal_fitted/total_fitted:.1f}%)")
    print(f"  ❌ Poor (R̂ ≥ 1.05):       {poor_fitted:>4d} ({100*poor_fitted/total_fitted:.1f}%)")
    
    max_rhat_fitted = fitted_summary['r_hat'].max()
    min_ess_fitted = fitted_summary['ess_bulk'].min()
    
    print(f"\nOverall fitted parameter stats:")
    print(f"  Max R̂: {max_rhat_fitted:.3f}")
    print(f"  Min ESS: {min_ess_fitted:.0f}")
    
    # Final assessment
    print(f"\n🎯 ASSESSMENT:")
    if poor_fitted == 0 and marginal_fitted < 0.1 * total_fitted:
        print("  ✅ GOOD: Most parameters converged well")
    elif poor_fitted < 0.05 * total_fitted and max_rhat_fitted < 1.1:
        print("  ⚠️  MARGINAL: Some convergence issues, interpret carefully")
    else:
        print("  ❌ POOR: Significant convergence problems - consider longer sampling")
    
    return fitted_summary


def print_parameter_correlations(trace, max_params=20):
    """
    Print correlations between key parameters
    """
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Get summary to find variable names
        summary = az.summary(trace)
        
        # Get core parameters only
        core_params = [v for v in summary.index if any(p in v for p in ['mu[', 'sigma[', 'tau', 'f0['])]
        
        if len(core_params) == 0:
            print("No core parameters found for correlation analysis")
            return
        
        print(f"\n🔄 PARAMETER CORRELATIONS")
        print("-"*60)
        print(f"Found {len(core_params)} core parameters for correlation analysis")
        
        # Take a subset if too many
        if len(core_params) > max_params:
            core_params = core_params[:max_params]
            print(f"Analyzing first {max_params} parameters")
        
        # Extract posterior samples for core parameters
        param_data = {}
        for param in core_params:
            try:
                # Navigate through the trace structure
                param_clean = param.replace('LLM_Factor_Model::', '')
                
                # Handle different parameter types
                if '[' in param_clean:
                    # Parse indexed parameters like mu[0], sigma[1], etc.
                    base_name = param_clean.split('[')[0]
                    index = int(param_clean.split('[')[1].split(']')[0])
                    
                    if base_name in trace.posterior.data_vars:
                        # Get the samples for this specific index
                        samples = trace.posterior[base_name].isel({f'{base_name}_dim_0': index}).values.flatten()
                        param_data[param_clean] = samples
                else:
                    # Handle scalar parameters like tau
                    if param_clean in trace.posterior.data_vars:
                        samples = trace.posterior[param_clean].values.flatten()
                        param_data[param_clean] = samples
                        
            except Exception as e:
                print(f"  Warning: Could not extract {param}: {e}")
                continue
        
        if len(param_data) < 2:
            print("Not enough parameters extracted for correlation analysis")
            return
        
        df = pd.DataFrame(param_data)
        corr_matrix = df.corr()
        
        print(f"Successfully extracted {len(df.columns)} parameters")
        
        # Show high correlations
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # High correlation threshold
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_val
                    ))
        
        if high_corr_pairs:
            print(f"\nHigh correlations (|r| > 0.5):")
            for param1, param2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]:
                print(f"  {param1:<12} ↔ {param2:<12}: {corr:>6.3f}")
        else:
            print("No high correlations found (all |r| ≤ 0.5)")
        
        # Show parameter ranges
        print(f"\nParameter ranges:")
        for col in df.columns:
            print(f"  {col:<12}: [{df[col].min():>7.3f}, {df[col].max():>7.3f}]")



# Analyze only the fitted parameters
fitted_summary = analyze_fitted_parameters(trace_vec, k=2)

# Optional: Check parameter correlations
print_parameter_correlations(trace_vec)
# %%
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_factor_parameters(trace, tp, g_sigmoid=0.0919):
    """
    Extract and print key factor model parameters
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        PyMC trace object with posterior samples
    tp : array
        Time points used in the model
    g_sigmoid : float
        Sigmoid steepness parameter used in model
    """
    
    print("="*70)
    print("FACTOR MODEL PARAMETER SUMMARY")
    print("="*70)
    
    # First, check what variables are available
    posterior = trace.posterior
    available_vars = list(posterior.data_vars.keys())
    print(f"Available variables in trace: {available_vars}")
    print()
    
    # Extract posterior means and stds
    
    # Try to extract parameters with error handling
    params = {}
    
    # Initial factor values (f0)
    f0_key = 'LLM_Factor_Model::f0'
    if f0_key in available_vars:
        f0_mean = posterior[f0_key].mean(dim=['chain', 'draw']).values
        f0_std = posterior[f0_key].std(dim=['chain', 'draw']).values
        params['f0'] = {'mean': f0_mean, 'std': f0_std}
        
        print("\n📍 INITIAL FACTOR VALUES (f₀)")
        print("-" * 40)
        for i, (mean, std) in enumerate(zip(f0_mean, f0_std)):
            sigmoid_val = 1 / (1 + np.exp(-g_sigmoid * mean))
            print(f"f₀[{i}]: {mean:7.2f} ± {std:5.2f} → sigmoid = {sigmoid_val:.1%}")
    else:
        print("\n⚠️  f0 variable not found in trace")
    
    # Drift parameters (mu) 
    mu_key = 'LLM_Factor_Model::mu'
    if mu_key in available_vars:
        mu_mean = posterior[mu_key].mean(dim=['chain', 'draw']).values
        mu_std = posterior[mu_key].std(dim=['chain', 'draw']).values
        params['mu'] = {'mean': mu_mean, 'std': mu_std}
        
        print("\n📈 DRIFT PARAMETERS (μ - daily improvement)")
        print("-" * 50)
        for i, (mean, std) in enumerate(zip(mu_mean, mu_std)):
            print(f"μ[{i}]: {mean:7.4f} ± {std:7.4f} ({mean*100:.2f}% per day)")
    else:
        print("\n⚠️  mu variable not found in trace")
    
    # Innovation variances (sigma)
    sigma_key = 'LLM_Factor_Model::sigma'
    if sigma_key in available_vars:
        sigma_mean = posterior[sigma_key].mean(dim=['chain', 'draw']).values  
        sigma_std = posterior[sigma_key].std(dim=['chain', 'draw']).values
        params['sigma'] = {'mean': sigma_mean, 'std': sigma_std}
        
        print("\n📊 INNOVATION VARIANCES (σ - daily volatility)")
        print("-" * 55)
        for i, (mean, std) in enumerate(zip(sigma_mean, sigma_std)):
            print(f"σ[{i}]: {mean:7.4f} ± {std:7.4f}")
    else:
        print("\n⚠️  sigma variable not found in trace")
    
    # Observation error
    tau_key = 'LLM_Factor_Model::tau'
    if tau_key in available_vars:
        tau_mean = posterior[tau_key].mean(dim=['chain', 'draw']).values
        tau_std = posterior[tau_key].std(dim=['chain', 'draw']).values
        params['tau'] = {'mean': tau_mean, 'std': tau_std}
        
        print(f"\n🎯 OBSERVATION ERROR")
        print("-" * 25)
        print(f"τ: {tau_mean:7.4f} ± {tau_std:7.4f}")
    else:
        print("\n⚠️  tau variable not found in trace")
    
    # Model comparison insights
    print(f"\n💡 KEY INSIGHTS")
    print("-" * 20)
    if 'mu' in params and len(params['mu']['mean']) >= 2:
        mu_mean = params['mu']['mean']
        ratio = mu_mean[1] / mu_mean[0] if mu_mean[0] > 0 else np.inf
        print(f"• Factor 2 improves {ratio:.1f}× faster than Factor 1")
        
    if 'sigma' in params and len(params['sigma']['mean']) >= 2:
        sigma_mean = params['sigma']['mean']
        vol_ratio = sigma_mean[1] / sigma_mean[0] if sigma_mean[0] > 0 else np.inf  
        print(f"• Factor 2 has {vol_ratio:.1f}× higher volatility than Factor 1")
    
    if 'tau' in params:
        tau_mean = params['tau']['mean']
        print(f"• Measurement noise: {tau_mean*100:.1f}% standard deviation")
    
    print(f"• Time span: {tp.max()} days ({tp.max()/365.25:.1f} years)")
    
    return params


def cluster_tasks_by_loadings(trace, tasks, n_clusters=2, plot=True, figsize=(12, 8)):
    """
    Cluster tasks based on factor loadings
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        PyMC trace object with posterior samples
    tasks : list
        List of task IDs corresponding to the loadings
    n_clusters : int
        Number of clusters (2 or 3 recommended)
    plot : bool
        Whether to create visualization plots
    figsize : tuple
        Figure size for plots
        
    Returns:
    --------
    pd.DataFrame : Task clustering results
    """
    
    # Check what loading variables are available
    posterior = trace.posterior
    available_vars = list(posterior.data_vars.keys())
    
    # Look for loading matrix variables
    loading_var = None
    possible_loading_vars = [
        'LLM_Factor_Model::loadings_matrix', 
        'LLM_Factor_Model::loadings_raw', 
        'LLM_Factor_Model::loadings',
        'loadings_matrix', 
        'loadings_raw', 
        'loadings'
    ]
    
    for var_name in possible_loading_vars:
        if var_name in available_vars:
            loading_var = var_name
            break
    
    if loading_var is None:
        print("❌ No loading variables found in trace!")
        print(f"Available variables: {available_vars}")
        return None
    
    # Extract factor loadings (posterior means)
    loadings = posterior[loading_var].mean(dim=['chain', 'draw']).values
    n_tasks, n_factors = loadings.shape
    
    print("="*70)
    print(f"TASK CLUSTERING BY FACTOR LOADINGS")
    print("="*70)
    print(f"Using variable: {loading_var}")
    print(f"Tasks: {n_tasks}, Factors: {n_factors}, Clusters: {n_clusters}")
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(loadings)
    
    # Create results dataframe
    cluster_df = pd.DataFrame({
        'task_id': tasks,
        'factor_1': loadings[:, 0],
        'factor_2': loadings[:, 1] if n_factors > 1 else 0,
        'cluster': clusters + 1  # 1-indexed clusters
    })
    
    if n_factors > 2:
        cluster_df['factor_3'] = loadings[:, 2]
    
    # Analyze clusters
    print(f"\n📊 CLUSTER SUMMARY")
    print("-" * 30)
    
    for cluster_id in range(1, n_clusters + 1):
        cluster_tasks = cluster_df[cluster_df['cluster'] == cluster_id]
        n_tasks_cluster = len(cluster_tasks)
        pct = n_tasks_cluster / len(cluster_df) * 100
        
        f1_mean = cluster_tasks['factor_1'].mean()
        f2_mean = cluster_tasks['factor_2'].mean()
        
        print(f"\nCluster {cluster_id}: {n_tasks_cluster} tasks ({pct:.1f}%)")
        print(f"  Avg Factor 1 loading: {f1_mean:.3f}")
        print(f"  Avg Factor 2 loading: {f2_mean:.3f}")
        
        # Interpretation
        if f1_mean > f2_mean and f1_mean > 0.5:
            interpretation = "High-Performance Tasks (Factor 1 dominant)"
        elif f2_mean > f1_mean and f2_mean > 0.5:
            interpretation = "Learning-Intensive Tasks (Factor 2 dominant)"  
        else:
            interpretation = "Mixed/Balanced Tasks"
            
        print(f"  Interpretation: {interpretation}")
        
        # Show a few example tasks
        example_tasks = cluster_tasks['task_id'].head(5).tolist()
        print(f"  Example tasks: {example_tasks}")
    
    # Create visualization if requested
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Scatter plot of loadings
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for cluster_id in range(1, n_clusters + 1):
            cluster_data = cluster_df[cluster_df['cluster'] == cluster_id]
            axes[0].scatter(cluster_data['factor_1'], cluster_data['factor_2'], 
                          c=colors[cluster_id-1], label=f'Cluster {cluster_id}', 
                          alpha=0.7, s=50)
        
        axes[0].set_xlabel('Factor 1 Loading (High Performance)')
        axes[0].set_ylabel('Factor 2 Loading (Learning)')
        axes[0].set_title('Task Clustering by Factor Loadings')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Bar plot of cluster sizes
        cluster_counts = cluster_df['cluster'].value_counts().sort_index()
        bars = axes[1].bar(cluster_counts.index, cluster_counts.values, 
                          color=[colors[i] for i in range(len(cluster_counts))])
        axes[1].set_xlabel('Cluster')
        axes[1].set_ylabel('Number of Tasks') 
        axes[1].set_title('Cluster Sizes')
        axes[1].grid(True, alpha=0.3)
        
        # Add count labels on bars
        for bar, count in zip(bars, cluster_counts.values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    # Sort by cluster for easier inspection
    cluster_df = cluster_df.sort_values(['cluster', 'factor_1'], ascending=[True, False])
    
    print(f"\n🎯 CLUSTERING COMPLETE")
    print(f"Returned dataframe with {len(cluster_df)} tasks and cluster assignments")
    
    return cluster_df


def plot_factor_trajectories(trace, tp, g_sigmoid=0.0919, figsize=(14, 5)):
    """
    Plot factor evolution over time
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        PyMC trace object with posterior samples
    tp : array
        Time points used in the model  
    g_sigmoid : float
        Sigmoid steepness parameter used in model
    figsize : tuple
        Figure size
    """
    
    # Check if factors variable exists
    posterior = trace.posterior
    available_vars = list(posterior.data_vars.keys())
    
    factors_key = 'LLM_Factor_Model::factors'
    if factors_key not in available_vars:
        print("❌ 'LLM_Factor_Model::factors' variable not found in trace!")
        print(f"Available variables: {available_vars}")
        return None
    
    # Extract factor trajectories
    factors = posterior[factors_key].mean(dim=['chain', 'draw']).values
    factors_std = posterior[factors_key].std(dim=['chain', 'draw']).values
    
    n_times, n_factors = factors.shape
    
    print(f"Plotting {n_factors} factors over {n_times} time points")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Plot 1: Raw factor values
    for i in range(n_factors):
        axes[0].plot(tp, factors[:, i], color=colors[i], linewidth=2.5, 
                    label=f'Factor {i+1}', marker='o', markersize=4)
        
        # Add confidence bands
        axes[0].fill_between(tp, 
                           factors[:, i] - 1.96 * factors_std[:, i],
                           factors[:, i] + 1.96 * factors_std[:, i],
                           color=colors[i], alpha=0.2)
    
    axes[0].set_xlabel('Days')
    axes[0].set_ylabel('Factor Value (Logits)')
    axes[0].set_title('Factor Evolution (Raw Values)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Sigmoid-transformed values
    for i in range(n_factors):
        sigmoid_values = 1 / (1 + np.exp(-g_sigmoid * factors[:, i]))
        sigmoid_lower = 1 / (1 + np.exp(-g_sigmoid * (factors[:, i] - 1.96 * factors_std[:, i])))
        sigmoid_upper = 1 / (1 + np.exp(-g_sigmoid * (factors[:, i] + 1.96 * factors_std[:, i])))
        
        axes[1].plot(tp, sigmoid_values, color=colors[i], linewidth=2.5,
                    label=f'Factor {i+1}', marker='o', markersize=4)
        
        # Add confidence bands
        axes[1].fill_between(tp, sigmoid_lower, sigmoid_upper,
                           color=colors[i], alpha=0.2)
    
    axes[1].set_xlabel('Days')
    axes[1].set_ylabel('Performance Score')
    axes[1].set_title('Factor Evolution (Sigmoid-Transformed)')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return factors


# Example usage:


#%%


# After running your model and getting the trace:

# Analyze parameters
params = analyze_factor_parameters(trace_vec, tp_vec)

# Cluster tasks 
cluster_results = cluster_tasks_by_loadings(trace_vec, tasks_vec, n_clusters=2)

# Plot factor trajectories
factors = plot_factor_trajectories(trace_vec, tp_vec)

# Access cluster results
if cluster_results is not None:
    print("\nFirst 10 clustered tasks:")
    print(cluster_results.head(10))
    
    # Save results
    cluster_results.to_csv('task_clusters.csv', index=False)


# %%
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def create_cluster_wordclouds(cluster_results, df, text_column='task_description', 
                             figsize=(15, 10), max_words=50, background_color='white',
                             use_tfidf=True, min_df=1, max_df=0.99):
    """
    Create word clouds for each cluster based on task descriptions using TF-IDF
    
    Parameters:
    -----------
    cluster_results : pd.DataFrame
        Results from cluster_tasks_by_loadings() with task_id and cluster columns
    df : pd.DataFrame  
        Original dataframe with task_id and task_description columns
    text_column : str
        Column name containing text descriptions (default: 'task_description')
    figsize : tuple
        Figure size for the plot
    max_words : int
        Maximum number of words in each word cloud
    background_color : str
        Background color for word clouds
    use_tfidf : bool
        Whether to use TF-IDF scoring (True) or raw frequency (False)
    min_df : int
        Minimum document frequency for TF-IDF (ignore words appearing in fewer clusters)
    max_df : float
        Maximum document frequency for TF-IDF (ignore words appearing in more than this fraction of clusters)
        
    Returns:
    --------
    dict : Cluster text analysis results
    """
    
    # Get unique tasks only (deduplicate df first)
    df_unique = df[['task_id', text_column]].drop_duplicates(subset=['task_id'])
    
    print(f"🔍 DATA VERIFICATION:")
    print(f"  Original df: {len(df)} rows")
    print(f"  Unique tasks in df: {len(df_unique)}")
    print(f"  Cluster results: {len(cluster_results)} tasks")
    
    # Merge cluster results with unique task descriptions
    merged_df = cluster_results.merge(df_unique, on='task_id', how='left')
    
    print(f"  After merge: {len(merged_df)} tasks")
    print(f"  Tasks per cluster: {merged_df['cluster'].value_counts().sort_index().to_dict()}")
    
    # Verify we have exactly the expected number of tasks
    if len(merged_df) != len(cluster_results):
        print(f"⚠️  WARNING: Merge resulted in {len(merged_df)} tasks but expected {len(cluster_results)}")
        
    if len(cluster_results) != 99:
        print(f"⚠️  WARNING: cluster_results has {len(cluster_results)} tasks but expected 99")
    
    # Check for missing descriptions
    missing_descriptions = merged_df[text_column].isna().sum()
    if missing_descriptions > 0:
        print(f"⚠️  Warning: {missing_descriptions} tasks have missing descriptions")
        merged_df = merged_df.dropna(subset=[text_column])
    
    n_clusters = cluster_results['cluster'].nunique()
    clusters = sorted(cluster_results['cluster'].unique())
    
    print("="*70)
    print("CLUSTER WORD CLOUD ANALYSIS" + (" (TF-IDF)" if use_tfidf else " (Frequency)"))
    print("="*70)
    print(f"Clusters: {n_clusters}, Unique tasks analyzed: {len(merged_df)}")
    
    # Set up extended stopwords
    stop_words = set(stopwords.words('english'))
    # Add domain-specific stopwords
    custom_stopwords = {
        'task', 'tasks', 'work', 'job', 'jobs', 'may', 'include', 'including', 
        'use', 'using', 'used', 'one', 'two', 'three', 'also', 'would', 'could',
        'will', 'shall', 'must', 'need', 'needs', 'required', 'require',
        'perform', 'performs', 'performing', 'conduct', 'conducts', 'conducting',
        'provide', 'provides', 'providing', 'ensure', 'ensures', 'ensuring',
        'develop', 'develops', 'developing', 'create', 'creates', 'creating',
        'maintain', 'maintains', 'maintaining', 'manage', 'manages', 'managing',
        'review', 'reviews', 'reviewing', 'analyze', 'analyzes', 'analyzing',
        'prepare', 'prepares', 'preparing', 'assist', 'assists', 'assisting',
        'coordinate', 'coordinates', 'coordinating', 'monitor', 'monitors', 'monitoring',
        'evaluate', 'evaluates', 'evaluating', 'implement', 'implements', 'implementing',
        'various', 'appropriate', 'necessary', 'effective', 'efficient', 'proper'
    }
    stop_words.update(custom_stopwords)
    
    def clean_text(text):
        """Clean and preprocess text for analysis"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        cleaned_tokens = [word for word in tokens 
                         if word not in stop_words and len(word) > 2]
        
        return ' '.join(cleaned_tokens)
    
    # Prepare cluster documents for TF-IDF
    cluster_documents = []
    cluster_info = {}
    
    for cluster_id in clusters:
        cluster_data = merged_df[merged_df['cluster'] == cluster_id]
        combined_text = ' '.join(cluster_data[text_column].fillna('').astype(str))
        cleaned_text = clean_text(combined_text)
        
        cluster_documents.append(cleaned_text)
        cluster_info[cluster_id] = {
            'n_tasks': len(cluster_data),
            'cleaned_text': cleaned_text
        }
    
    # Apply TF-IDF if requested
    if use_tfidf and len(cluster_documents) > 1:
        print(f"\n🔬 TF-IDF COMPUTATION:")
        print(f"  Computing TF-IDF with min_df={min_df}, max_df={max_df}")
        print(f"  Number of cluster documents: {len(cluster_documents)}")
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            min_df=min_df,  # Words must appear in at least min_df clusters
            max_df=max_df,  # Only remove words appearing in >99% of clusters
            ngram_range=(1, 1),  # Only single words
            max_features=None
        )
        
        # Fit TF-IDF on all cluster documents
        tfidf_matrix = vectorizer.fit_transform(cluster_documents)
        feature_names = vectorizer.get_feature_names_out()
        
        print(f"  TF-IDF vocabulary size: {len(feature_names)}")
        print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
    else:
        print(f"\n📊 Using frequency-based analysis (not TF-IDF)")
    
    # Analyze each cluster
    cluster_analysis = {}
    
    # Create subplots for word clouds + scatter plot
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Define consistent colors for clusters
    cluster_colors = {
        1: '#1f77b4',  # Blue
        2: '#ff7f0e',  # Orange  
        3: '#2ca02c',  # Green
        4: '#d62728',  # Red
        5: '#9467bd'   # Purple
    }
    
    # Colormap names for word clouds (darker, more readable)
    colormaps = ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples']
    
    for i, cluster_id in enumerate(clusters):
        n_tasks = cluster_info[cluster_id]['n_tasks']
        cleaned_text = cluster_info[cluster_id]['cleaned_text']
        
        print(f"\n📊 CLUSTER {cluster_id} ({n_tasks} tasks)")
        print("-" * 40)
        
        if not cleaned_text.strip():
            print("⚠️  No valid text found for this cluster")
            cluster_analysis[cluster_id] = {
                'n_tasks': n_tasks,
                'top_words': [],
                'text_length': 0
            }
            continue
        
        # Get word importance scores
        if use_tfidf and len(cluster_documents) > 1:
            # Get TF-IDF scores for this cluster
            cluster_tfidf = tfidf_matrix[i].toarray()[0]
            
            # Create word-score pairs
            word_scores = list(zip(feature_names, cluster_tfidf))
            # Sort by TF-IDF score (descending)
            word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
            # Filter out zero scores
            word_scores = [(word, score) for word, score in word_scores if score > 0]
            
            top_words = word_scores[:20]
            print(f"Top TF-IDF words: {[f'{word}({score:.3f})' for word, score in top_words[:8]]}")
            
            # Create word frequency dict for WordCloud (normalize scores to reasonable range)
            if word_scores:
                max_score = max(score for word, score in word_scores)
                # Scale scores to 1-1000 range for better WordCloud visualization
                word_freq_dict = {word: int(score * 1000 / max_score) + 1 
                                for word, score in word_scores if score > 0}
                print(f"WordCloud dict sample: {dict(list(word_freq_dict.items())[:5])}")
            else:
                word_freq_dict = {}

        else:
            # Fallback to regular frequency
            word_counts = Counter(cleaned_text.split())
            top_words = word_counts.most_common(20)
            print(f"Top frequency words: {[f'{word}({count})' for word, count in top_words[:8]]}")
            word_freq_dict = dict(word_counts)
        
        # Create word cloud (only for first 3 clusters, save 4th spot for scatter plot)
        if i < 3 and word_freq_dict:
            print(f"Creating wordcloud with {len(word_freq_dict)} words")
            
            # Define custom darker color functions for better visibility
            def color_func_blue(word, font_size, position, orientation, random_state=None, **kwargs):
                return f"hsl(210, 80%, {20 + font_size/100*30}%)"
            
            def color_func_orange(word, font_size, position, orientation, random_state=None, **kwargs):
                return f"hsl(30, 90%, {25 + font_size/100*35}%)"
            
            def color_func_green(word, font_size, position, orientation, random_state=None, **kwargs):
                return f"hsl(120, 70%, {20 + font_size/100*30}%)"
            
            color_funcs = [color_func_blue, color_func_orange, color_func_green]
            
            wordcloud = WordCloud(
                width=400, 
                height=300,
                background_color=background_color,
                max_words=max_words,  # Reduced from 100 to 50
                color_func=color_funcs[i],  # Custom dark colors
                relative_scaling=0.8,       # Increased for more size variation
                font_step=1,                # Smaller step for smoother sizes
                min_font_size=12,           # Larger minimum font (was 8)
                max_font_size=80,           # Larger maximum font (was 60)
                prefer_horizontal=0.8,      # More horizontal text (easier to read)
                random_state=42,
                collocations=False,         # Avoid word pairs
                margin=5                    # Smaller margin for more space
            ).generate_from_frequencies(word_freq_dict)
            
            axes[i].imshow(wordcloud, interpolation='bilinear')
            method = "TF-IDF" if use_tfidf else "Frequency"
            axes[i].set_title(f'Cluster {cluster_id} ({method})\n({n_tasks} tasks)', 
                            fontsize=14, fontweight='bold')
            axes[i].axis('off')
        elif i < 3:
            print(f"⚠️  No words available for wordcloud in cluster {cluster_id}")
        
        # Store analysis results
        cluster_analysis[cluster_id] = {
            'n_tasks': n_tasks,
            'top_words': top_words,
            'text_length': len(cleaned_text.split()),
            'sample_descriptions': merged_df[merged_df['cluster'] == cluster_id][text_column].head(3).tolist()
        }
    
    # Create scatter plot in the 4th position (index 3)
    if len(axes) > 3:
        ax_scatter = axes[3]
        
        # Check if we have factor loading data
        if 'factor_1' in cluster_results.columns and 'factor_2' in cluster_results.columns:
            
            # Plot each cluster with its color
            for cluster_id in clusters:
                cluster_data = cluster_results[cluster_results['cluster'] == cluster_id]
                
                ax_scatter.scatter(
                    cluster_data['factor_1'], 
                    cluster_data['factor_2'],
                    c=cluster_colors.get(cluster_id, '#gray'),
                    label=f'Cluster {cluster_id} (n={len(cluster_data)})',
                    alpha=0.7,
                    s=50,
                    edgecolors='white',
                    linewidth=0.5
                )
            
            ax_scatter.set_xlabel('Factor 1 Loading (High Performance)', fontsize=12)
            ax_scatter.set_ylabel('Factor 2 Loading (Learning)', fontsize=12)
            ax_scatter.set_title('Task Clustering in Factor Space', fontsize=14, fontweight='bold')
            ax_scatter.legend(fontsize=10)
            ax_scatter.grid(True, alpha=0.3)
            
            # Add cluster centroids
            for cluster_id in clusters:
                cluster_data = cluster_results[cluster_results['cluster'] == cluster_id]
                centroid_x = cluster_data['factor_1'].mean()
                centroid_y = cluster_data['factor_2'].mean()
                
                ax_scatter.scatter(
                    centroid_x, centroid_y,
                    c=cluster_colors.get(cluster_id, '#gray'),
                    s=200,
                    marker='x',
                    linewidth=3,
                    edgecolors='black'
                )
        else:
            ax_scatter.text(0.5, 0.5, 'Factor loading data\nnot available', 
                          ha='center', va='center', transform=ax_scatter.transAxes,
                          fontsize=12)
            ax_scatter.set_title('Factor Loadings', fontsize=14, fontweight='bold')
        
        ax_scatter.axis('on')  # Keep axes for scatter plot
    
    # Hide any remaining unused subplots
    for j in range(len(clusters) + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    method_title = "TF-IDF" if use_tfidf else "Frequency"
    plt.suptitle(f'Cluster Analysis: Word Clouds ({method_title}) & Factor Space', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.show()
    
    # Final verification
    print(f"\n✅ FINAL VERIFICATION:")
    total_tasks_analyzed = sum(analysis.get('n_tasks', 0) for analysis in cluster_analysis.values())
    print(f"  Total tasks analyzed: {total_tasks_analyzed}")
    print(f"  Expected tasks: {len(cluster_results)}")
    print(f"  Using method: {'TF-IDF' if use_tfidf else 'Frequency'}")
    
    if total_tasks_analyzed != len(cluster_results):
        print(f"⚠️  MISMATCH: Analyzed {total_tasks_analyzed} but expected {len(cluster_results)}")
    else:
        print("✅ Task count verified correctly")
    
    # Print detailed analysis
    print(f"\n💡 CLUSTER INTERPRETATION")
    print("="*50)
    
    for cluster_id, analysis in cluster_analysis.items():
        if analysis['top_words']:
            if use_tfidf:
                top_5_words = [word for word, score in analysis['top_words'][:5]]
            else:
                top_5_words = [word for word, count in analysis['top_words'][:5]]
            print(f"\nCluster {cluster_id}: {' • '.join(top_5_words)}")
            print(f"  Tasks: {analysis['n_tasks']}")
            print(f"  Theme: {interpret_cluster_theme(top_5_words)}")
    
    return cluster_analysis

def interpret_cluster_theme(top_words):
    """Provide interpretation based on top words"""
    words_str = ' '.join(top_words).lower()
    
    # Define theme keywords
    themes = {
        'management': ['manage', 'supervise', 'direct', 'oversee', 'coordinate', 'lead', 'staff', 'team'],
        'analysis': ['analyze', 'data', 'research', 'evaluate', 'assess', 'study', 'examine', 'investigate'],
        'technical': ['technical', 'system', 'software', 'computer', 'equipment', 'technology', 'engineering'],
        'financial': ['financial', 'budget', 'cost', 'money', 'economic', 'accounting', 'revenue', 'profit'],
        'communication': ['communicate', 'present', 'report', 'write', 'document', 'meeting', 'discuss'],
        'healthcare': ['patient', 'medical', 'health', 'care', 'treatment', 'clinical', 'therapy'],
        'education': ['teach', 'student', 'education', 'learn', 'training', 'curriculum', 'instruction'],
        'sales': ['sell', 'customer', 'client', 'market', 'product', 'service', 'sales'],
        'operations': ['operate', 'process', 'procedure', 'production', 'quality', 'safety', 'maintenance']
    }
    
    # Score each theme
    theme_scores = {}
    for theme, keywords in themes.items():
        score = sum(1 for keyword in keywords if keyword in words_str)
        if score > 0:
            theme_scores[theme] = score
    
    if theme_scores:
        best_theme = max(theme_scores, key=theme_scores.get)
        return f"{best_theme.title()} & Operations"
    else:
        return "General Work Tasks"

def print_cluster_details(cluster_results, df, text_column='task_description'):
    """Print detailed breakdown of tasks per cluster"""
    
    # Get unique tasks only (deduplicate df first)
    df_unique = df[['task_id', text_column, 'occupation']].drop_duplicates(subset=['task_id'])
    
    merged_df = cluster_results.merge(df_unique, on='task_id', how='left')
    
    print("\n" + "="*70)
    print("DETAILED CLUSTER BREAKDOWN")
    print("="*70)
    
    for cluster_id in sorted(cluster_results['cluster'].unique()):
        cluster_data = merged_df[merged_df['cluster'] == cluster_id]
        
        print(f"\n🔍 CLUSTER {cluster_id} - {len(cluster_data)} tasks")
        print("-" * 50)
        
        # Most common occupations
        if 'occupation' in cluster_data.columns:
            top_occupations = cluster_data['occupation'].value_counts().head(5)
            print("Top occupations:")
            for occ, count in top_occupations.items():
                print(f"  • {occ}: {count} tasks")
        
        print("\nExample task descriptions:")
        for i, (_, row) in enumerate(cluster_data.head(3).iterrows()):
            desc = row[text_column]
            if pd.notna(desc):
                desc_short = (desc[:100] + "...") if len(desc) > 100 else desc
                print(f"  {i+1}. {desc_short}")

# Example usage:
"""
# TF-IDF word clouds (optimized for 3 clusters):
cluster_analysis = create_cluster_wordclouds(cluster_results, df, use_tfidf=True)

# For more aggressive filtering (only if you want very distinctive words):
cluster_analysis = create_cluster_wordclouds(
    cluster_results, df, use_tfidf=True, 
    min_df=1, max_df=0.67  # Remove words in 2+ of 3 clusters
)

# Print detailed breakdown:
print_cluster_details(cluster_results, df)
"""
#%%

# TF-IDF word clouds (optimized for 3 clusters):
cluster_analysis = create_cluster_wordclouds(cluster_results, df, use_tfidf=True)

# For more aggressive filtering (only if you want very distinctive words):
cluster_analysis = create_cluster_wordclouds(
    cluster_results, df, use_tfidf=True, 
    min_df=2, max_df=0.6  # Remove words in 2+ of 3 clusters
)

# Print detailed breakdown:
print
#%%
