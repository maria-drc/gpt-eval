import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from typing import Optional, List, Union

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
                      sigma_prior=0.1,            # beta for HalfCauchy  
                      tau_prior=0.05,             # beta for HalfCauchy
                      f0_prior=(-10, 10),         # (mean, std) for initial factors
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
            2: [18868, 1094],
            3: [18868, 1094, 15735]
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
        print(f"  f0_prior: N({f0_prior[0]}, {f0_prior[1]})")
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
        f0 = pm.Normal('f0', mu=f0_prior[0], sigma=f0_prior[1], shape=k, 
                      initval=np.full(k, f0_prior[0]))
        
        print("Setting up factor evolution...")
        
        # Factor evolution: implement random walk with uneven time steps
        # f[0] ~ N(f0, small_variance) and f[t] = f[t-1] + N(Δt·μ, Δt·σ²)
        f_list = [f0]  # Start with initial values
        
        for t in range(1, T):
            dt = time_diffs[t]  # Time difference from previous observation
            f_prev = f_list[-1]
            
            # Increment ~ N(dt·μ, dt·σ²) for each factor
            f_increment = pm.Normal(f'f_increment_{t}', 
                                   mu=dt * mu, 
                                   sigma=pt.sqrt(dt) * sigma, 
                                   shape=k)
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

# Comprehensive sampling function (no diagnostics)
def fit_factor_model(model, draws=1000, tune=2000, chains=4, 
                    target_accept=0.95, cores=4):
    """
    Fit the factor model with MCMC sampling
    
    Parameters:
    -----------
    model : PyMC model
        The built factor model
    draws : int
        Number of posterior samples per chain
    tune : int  
        Number of tuning samples per chain
    chains : int
        Number of MCMC chains
    target_accept : float
        Target acceptance probability (higher = more conservative)
    cores : int
        Number of CPU cores for parallel sampling
    
    Returns:
    --------
    trace : ArviZ InferenceData
        MCMC trace with samples
    """
    
    print("="*80)
    print("FITTING DYNAMIC FACTOR MODEL")
    print("="*80)
    print(f"Chains: {chains}, Draws: {draws}, Tune: {tune}")
    print(f"Target acceptance: {target_accept}")
    
    # Sample from posterior
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            cores=cores,
            return_inferencedata=True,
            random_seed=42,
            idata_kwargs={'log_likelihood': True}  # For model comparison
        )
    
    print("✅ Sampling completed!")
    return trace


def analyze_trace_diagnostics(trace):
    """
    Comprehensive sampling and convergence diagnostics
    
    Parameters:
    -----------
    trace : ArviZ InferenceData
        MCMC trace from sampling
        
    Returns:
    --------
    dict : Dictionary with diagnostic results
    """
    
    print("\n" + "="*80)
    print("SAMPLING DIAGNOSTICS")
    print("="*80)
    
    diagnostics = {}
    
    # Basic sampling stats
    n_divergences = int(trace.sample_stats.diverging.sum().values)
    n_max_treedepth = int((trace.sample_stats.tree_depth >= 10).sum().values)
    
    print(f"Total divergences: {n_divergences}")
    print(f"Max tree depth hits: {n_max_treedepth}")
    
    diagnostics['n_divergences'] = n_divergences
    diagnostics['n_max_treedepth'] = n_max_treedepth
    
    if n_divergences > 0:
        print("⚠️  Divergences detected - consider increasing target_accept or reparameterizing")
    else:
        print("✅ No divergences")
    
    # Convergence diagnostics
    print("\n" + "-"*60)
    print("CONVERGENCE DIAGNOSTICS")
    print("-"*60)
    
    # Get summary statistics
    summary = az.summary(trace)
    
    # R-hat diagnostics
    rhat_values = summary['r_hat']
    rhat_max = float(rhat_values.max())
    rhat_bad = int((rhat_values > 1.01).sum())
    
    print(f"R-hat statistics:")
    print(f"  Max R-hat: {rhat_max:.4f}")
    print(f"  Parameters with R-hat > 1.01: {rhat_bad}/{len(rhat_values)}")
    
    diagnostics['rhat_max'] = rhat_max
    diagnostics['rhat_bad'] = rhat_bad
    
    if rhat_max < 1.01:
        print("✅ All parameters converged (R-hat < 1.01)")
        diagnostics['convergence'] = 'excellent'
    elif rhat_max < 1.05:
        print("⚠️  Marginal convergence (1.01 < R-hat < 1.05)")
        diagnostics['convergence'] = 'marginal'
    else:
        print("❌ Poor convergence (R-hat > 1.05) - need more samples")
        diagnostics['convergence'] = 'poor'
    
    # ESS diagnostics  
    ess_bulk = summary['ess_bulk']
    ess_tail = summary['ess_tail']
    ess_bulk_min = float(ess_bulk.min())
    ess_tail_min = float(ess_tail.min())
    ess_bulk_bad = int((ess_bulk < 400).sum())
    ess_tail_bad = int((ess_tail < 400).sum())
    
    print(f"\nEffective Sample Size (ESS) statistics:")
    print(f"  Min ESS (bulk): {ess_bulk_min:.0f}")
    print(f"  Min ESS (tail): {ess_tail_min:.0f}")
    print(f"  Parameters with ESS bulk < 400: {ess_bulk_bad}/{len(ess_bulk)}")
    print(f"  Parameters with ESS tail < 400: {ess_tail_bad}/{len(ess_tail)}")
    
    diagnostics['ess_bulk_min'] = ess_bulk_min
    diagnostics['ess_tail_min'] = ess_tail_min
    diagnostics['ess_bulk_bad'] = ess_bulk_bad
    diagnostics['ess_tail_bad'] = ess_tail_bad
    
    if ess_bulk_min > 400 and ess_tail_min > 400:
        print("✅ Good effective sample sizes")
        diagnostics['ess_quality'] = 'good'
    else:
        print("⚠️  Low effective sample sizes - consider more draws")
        diagnostics['ess_quality'] = 'poor'
    
    # Overall assessment
    print("\n" + "="*80)
    print("OVERALL SAMPLING ASSESSMENT")
    print("="*80)
    
    if n_divergences == 0 and rhat_max < 1.01 and ess_bulk_min > 400:
        print("🎉 EXCELLENT: Model converged well with good diagnostics!")
        diagnostics['overall'] = 'excellent'
    elif n_divergences < 20 and rhat_max < 1.05 and ess_bulk_min > 200:
        print("✅ GOOD: Model converged adequately - results should be reliable")
        diagnostics['overall'] = 'good'
    elif rhat_max < 1.1:
        print("⚠️  MARGINAL: Model may have converged - interpret results carefully")
        diagnostics['overall'] = 'marginal'
    else:
        print("❌ POOR: Model did not converge - need to rerun with more samples")
        diagnostics['overall'] = 'poor'
    
    return diagnostics


def analyze_factor_parameters(trace, k=3):
    """
    Analyze and interpret factor model parameters
    
    Parameters:
    -----------
    trace : ArviZ InferenceData
        MCMC trace from sampling
    k : int
        Number of factors
        
    Returns:
    --------
    dict : Dictionary with parameter estimates
    """
    
    print("\n" + "="*80)
    print("FACTOR MODEL PARAMETER ESTIMATES")
    print("="*80)
    
    # Get summary statistics
    summary = az.summary(trace)
    params = {}
    
    # Drift parameters (mu)
    print("\nDrift parameters (μ - daily improvement):")
    mu_estimates = []
    for i in range(k):
        mu_row = summary.loc[f'mu[{i}]']
        mu_mean = mu_row['mean']
        mu_std = mu_row['sd']
        mu_hdi_low = mu_row['hdi_3%']
        mu_hdi_high = mu_row['hdi_97%']
        mu_rhat = mu_row['r_hat']
        
        print(f"  Factor {i+1}: {mu_mean:.4f} ± {mu_std:.4f} "
              f"[{mu_hdi_low:.4f}, {mu_hdi_high:.4f}] "
              f"(R̂={mu_rhat:.3f})")
        
        mu_estimates.append({
            'mean': mu_mean, 'std': mu_std, 
            'hdi_low': mu_hdi_low, 'hdi_high': mu_hdi_high,
            'rhat': mu_rhat
        })
    
    params['mu'] = mu_estimates
    
    # Innovation variances (sigma)
    print("\nInnovation variances (σ - daily volatility):")
    sigma_estimates = []
    for i in range(k):
        sigma_row = summary.loc[f'sigma[{i}]']
        sigma_mean = sigma_row['mean']
        sigma_std = sigma_row['sd']
        sigma_hdi_low = sigma_row['hdi_3%']
        sigma_hdi_high = sigma_row['hdi_97%']
        sigma_rhat = sigma_row['r_hat']
        
        print(f"  Factor {i+1}: {sigma_mean:.4f} ± {sigma_std:.4f} "
              f"[{sigma_hdi_low:.4f}, {sigma_hdi_high:.4f}] "
              f"(R̂={sigma_rhat:.3f})")
        
        sigma_estimates.append({
            'mean': sigma_mean, 'std': sigma_std,
            'hdi_low': sigma_hdi_low, 'hdi_high': sigma_hdi_high,
            'rhat': sigma_rhat
        })
    
    params['sigma'] = sigma_estimates
    
    # Observation error (tau)
    tau_row = summary.loc['tau']
    tau_estimates = {
        'mean': tau_row['mean'], 'std': tau_row['sd'],
        'hdi_low': tau_row['hdi_3%'], 'hdi_high': tau_row['hdi_97%'],
        'rhat': tau_row['r_hat']
    }
    
    print(f"\nObservation error (τ):")
    print(f"  {tau_estimates['mean']:.4f} ± {tau_estimates['std']:.4f} "
          f"[{tau_estimates['hdi_low']:.4f}, {tau_estimates['hdi_high']:.4f}] "
          f"(R̂={tau_estimates['rhat']:.3f})")
    
    params['tau'] = tau_estimates
    
    # Initial factor values
    print(f"\nInitial factor values (f₀):")
    f0_estimates = []
    for i in range(k):
        f0_row = summary.loc[f'f0[{i}]']
        f0_mean = f0_row['mean']
        f0_std = f0_row['sd']
        f0_rhat = f0_row['r_hat']
        
        print(f"  Factor {i+1}: {f0_mean:.3f} ± {f0_std:.3f} "
              f"(R̂={f0_rhat:.3f})")
        
        f0_estimates.append({
            'mean': f0_mean, 'std': f0_std, 'rhat': f0_rhat
        })
    
    params['f0'] = f0_estimates
    
    return params


def analyze_factor_loadings(trace, task_ids, k=3, top_n=5):
    """
    Analyze and interpret factor loadings
    
    Parameters:
    -----------
    trace : ArviZ InferenceData
        MCMC trace from sampling
    task_ids : list
        List of task IDs corresponding to rows in loadings matrix
    k : int
        Number of factors
    top_n : int
        Number of top tasks to show per factor
        
    Returns:
    --------
    dict : Dictionary with loading analysis
    """
    
    print("\n" + "="*80)
    print("FACTOR LOADINGS ANALYSIS")
    print("="*80)
    
    # Extract posterior means for loadings
    loadings_mean = trace.posterior['loadings_matrix'].mean(dim=['chain', 'draw']).values
    loadings_std = trace.posterior['loadings_matrix'].std(dim=['chain', 'draw']).values
    
    # Get summary for convergence info
    summary = az.summary(trace)
    loadings_summary = summary.loc[summary.index.str.contains('loadings_raw')]
    
    print(f"Factor loadings summary statistics:")
    print(f"  Mean loading: {loadings_mean.mean():.4f}")
    print(f"  Loading range: [{loadings_mean.min():.4f}, {loadings_mean.max():.4f}]")
    if len(loadings_summary) > 0:
        print(f"  Worst R̂: {loadings_summary['r_hat'].max():.3f}")
        print(f"  Min ESS: {loadings_summary['ess_bulk'].min():.0f}")
    
    # Factor interpretation
    factor_analysis = {}
    
    print(f"\nTop {top_n} tasks per factor (by absolute loading):")
    for factor_idx in range(k):
        print(f"\n  === Factor {factor_idx + 1} ===")
        
        # Get loadings for this factor
        factor_loadings = loadings_mean[:, factor_idx]
        factor_loadings_std = loadings_std[:, factor_idx]
        
        # Sort by absolute loading value
        sorted_indices = np.argsort(np.abs(factor_loadings))[::-1]
        
        top_tasks = []
        for rank in range(min(top_n, len(task_ids))):
            task_idx = sorted_indices[rank]
            task_id = task_ids[task_idx]
            loading_val = factor_loadings[task_idx]
            loading_unc = factor_loadings_std[task_idx]
            
            print(f"    {rank+1}. Task {task_id}: {loading_val:.3f} ± {loading_unc:.3f}")
            
            top_tasks.append({
                'task_id': task_id,
                'task_idx': task_idx,
                'loading_mean': loading_val,
                'loading_std': loading_unc,
                'rank': rank + 1
            })
        
        factor_analysis[f'factor_{factor_idx + 1}'] = {
            'top_tasks': top_tasks,
            'mean_loading': float(factor_loadings.mean()),
            'std_loading': float(factor_loadings.std())
        }
    
    return factor_analysis, loadings_mean, loadings_std


def analyze_factor_evolution(trace, time_points, k=3):
    """
    Analyze factor evolution over time
    
    Parameters:
    -----------
    trace : ArviZ InferenceData
        MCMC trace from sampling
    time_points : array
        Time points (days from start)
    k : int
        Number of factors
        
    Returns:
    --------
    dict : Dictionary with factor evolution
    """
    
    print("\n" + "="*80)
    print("FACTOR EVOLUTION OVER TIME")
    print("="*80)
    
    # Extract factor values
    factors_mean = trace.posterior['factors'].mean(dim=['chain', 'draw']).values
    factors_std = trace.posterior['factors'].std(dim=['chain', 'draw']).values
    
    print("Factor values at key time points:")
    print(f"{'Time (days)':<12} ", end="")
    for i in range(k):
        print(f"Factor {i+1:<8}", end="  ")
    print()
    print("-" * (12 + k * 10))
    
    evolution_data = {}
    for t_idx, t in enumerate(time_points):
        print(f"{t:<12.0f} ", end="")
        
        factor_values_at_t = {}
        for factor_idx in range(k):
            factor_val = factors_mean[t_idx, factor_idx]
            factor_unc = factors_std[t_idx, factor_idx]
            print(f"{factor_val:>6.2f}±{factor_unc:.2f} ", end="")
            
            factor_values_at_t[f'factor_{factor_idx + 1}'] = {
                'mean': float(factor_val),
                'std': float(factor_unc)
            }
        
        evolution_data[f'time_{t}'] = factor_values_at_t
        print()
    
    # Calculate total improvement per factor
    print(f"\nTotal factor improvement (first to last time point):")
    for factor_idx in range(k):
        initial_val = factors_mean[0, factor_idx]
        final_val = factors_mean[-1, factor_idx]
        total_improvement = final_val - initial_val
        
        print(f"  Factor {factor_idx + 1}: {initial_val:.3f} → {final_val:.3f} "
              f"(Δ = {total_improvement:+.3f})")
    
    return evolution_data, factors_mean, factors_std

def analyze_trace_diagnostics(trace):
    """
    Comprehensive sampling and convergence diagnostics
    
    Parameters:
    -----------
    trace : ArviZ InferenceData
        MCMC trace from sampling
        
    Returns:
    --------
    dict : Dictionary with diagnostic results
    """
    
    print("\n" + "="*80)
    print("SAMPLING DIAGNOSTICS")
    print("="*80)
    
    diagnostics = {}
    
    # Basic sampling stats
    n_divergences = int(trace.sample_stats.diverging.sum().values)
    n_max_treedepth = int((trace.sample_stats.tree_depth >= 10).sum().values)
    
    print(f"Total divergences: {n_divergences}")
    print(f"Max tree depth hits: {n_max_treedepth}")
    
    diagnostics['n_divergences'] = n_divergences
    diagnostics['n_max_treedepth'] = n_max_treedepth
    
    if n_divergences > 0:
        print("⚠️  Divergences detected - consider increasing target_accept or reparameterizing")
    else:
        print("✅ No divergences")
    
    # Convergence diagnostics
    print("\n" + "-"*60)
    print("CONVERGENCE DIAGNOSTICS")
    print("-"*60)
    
    # Get summary statistics
    summary = az.summary(trace)
    
    # R-hat diagnostics
    rhat_values = summary['r_hat']
    rhat_max = float(rhat_values.max())
    rhat_bad = int((rhat_values > 1.01).sum())
    
    print(f"R-hat statistics:")
    print(f"  Max R-hat: {rhat_max:.4f}")
    print(f"  Parameters with R-hat > 1.01: {rhat_bad}/{len(rhat_values)}")
    
    diagnostics['rhat_max'] = rhat_max
    diagnostics['rhat_bad'] = rhat_bad
    
    if rhat_max < 1.01:
        print("✅ All parameters converged (R-hat < 1.01)")
        diagnostics['convergence'] = 'excellent'
    elif rhat_max < 1.05:
        print("⚠️  Marginal convergence (1.01 < R-hat < 1.05)")
        diagnostics['convergence'] = 'marginal'
    else:
        print("❌ Poor convergence (R-hat > 1.05) - need more samples")
        diagnostics['convergence'] = 'poor'
    
    # ESS diagnostics  
    ess_bulk = summary['ess_bulk']
    ess_tail = summary['ess_tail']
    ess_bulk_min = float(ess_bulk.min())
    ess_tail_min = float(ess_tail.min())
    ess_bulk_bad = int((ess_bulk < 400).sum())
    ess_tail_bad = int((ess_tail < 400).sum())
    
    print(f"\nEffective Sample Size (ESS) statistics:")
    print(f"  Min ESS (bulk): {ess_bulk_min:.0f}")
    print(f"  Min ESS (tail): {ess_tail_min:.0f}")
    print(f"  Parameters with ESS bulk < 400: {ess_bulk_bad}/{len(ess_bulk)}")
    print(f"  Parameters with ESS tail < 400: {ess_tail_bad}/{len(ess_tail)}")
    
    diagnostics['ess_bulk_min'] = ess_bulk_min
    diagnostics['ess_tail_min'] = ess_tail_min
    diagnostics['ess_bulk_bad'] = ess_bulk_bad
    diagnostics['ess_tail_bad'] = ess_tail_bad
    
    if ess_bulk_min > 400 and ess_tail_min > 400:
        print("✅ Good effective sample sizes")
        diagnostics['ess_quality'] = 'good'
    else:
        print("⚠️  Low effective sample sizes - consider more draws")
        diagnostics['ess_quality'] = 'poor'
    
    # Overall assessment
    print("\n" + "="*80)
    print("OVERALL SAMPLING ASSESSMENT")
    print("="*80)
    
    if n_divergences == 0 and rhat_max < 1.01 and ess_bulk_min > 400:
        print("🎉 EXCELLENT: Model converged well with good diagnostics!")
        diagnostics['overall'] = 'excellent'
    elif n_divergences < 20 and rhat_max < 1.05 and ess_bulk_min > 200:
        print("✅ GOOD: Model converged adequately - results should be reliable")
        diagnostics['overall'] = 'good'
    elif rhat_max < 1.1:
        print("⚠️  MARGINAL: Model may have converged - interpret results carefully")
        diagnostics['overall'] = 'marginal'
    else:
        print("❌ POOR: Model did not converge - need to rerun with more samples")
        diagnostics['overall'] = 'poor'
    
    return diagnostics


def analyze_factor_parameters(trace, k=3):
    """
    Analyze and interpret factor model parameters
    
    Parameters:
    -----------
    trace : ArviZ InferenceData
        MCMC trace from sampling
    k : int
        Number of factors
        
    Returns:
    --------
    dict : Dictionary with parameter estimates
    """
    
    print("\n" + "="*80)
    print("FACTOR MODEL PARAMETER ESTIMATES")
    print("="*80)
    
    # Get summary statistics
    summary = az.summary(trace)
    params = {}
    
    # Drift parameters (mu)
    print("\nDrift parameters (μ - daily improvement):")
    mu_estimates = []
    for i in range(k):
        mu_row = summary.loc[f'mu[{i}]']
        mu_mean = mu_row['mean']
        mu_std = mu_row['sd']
        mu_hdi_low = mu_row['hdi_3%']
        mu_hdi_high = mu_row['hdi_97%']
        mu_rhat = mu_row['r_hat']
        
        print(f"  Factor {i+1}: {mu_mean:.4f} ± {mu_std:.4f} "
              f"[{mu_hdi_low:.4f}, {mu_hdi_high:.4f}] "
              f"(R̂={mu_rhat:.3f})")
        
        mu_estimates.append({
            'mean': mu_mean, 'std': mu_std, 
            'hdi_low': mu_hdi_low, 'hdi_high': mu_hdi_high,
            'rhat': mu_rhat
        })
    
    params['mu'] = mu_estimates
    
    # Innovation variances (sigma)
    print("\nInnovation variances (σ - daily volatility):")
    sigma_estimates = []
    for i in range(k):
        sigma_row = summary.loc[f'sigma[{i}]']
        sigma_mean = sigma_row['mean']
        sigma_std = sigma_row['sd']
        sigma_hdi_low = sigma_row['hdi_3%']
        sigma_hdi_high = sigma_row['hdi_97%']
        sigma_rhat = sigma_row['r_hat']
        
        print(f"  Factor {i+1}: {sigma_mean:.4f} ± {sigma_std:.4f} "
              f"[{sigma_hdi_low:.4f}, {sigma_hdi_high:.4f}] "
              f"(R̂={sigma_rhat:.3f})")
        
        sigma_estimates.append({
            'mean': sigma_mean, 'std': sigma_std,
            'hdi_low': sigma_hdi_low, 'hdi_high': sigma_hdi_high,
            'rhat': sigma_rhat
        })
    
    params['sigma'] = sigma_estimates
    
    # Observation error (tau)
    tau_row = summary.loc['tau']
    tau_estimates = {
        'mean': tau_row['mean'], 'std': tau_row['sd'],
        'hdi_low': tau_row['hdi_3%'], 'hdi_high': tau_row['hdi_97%'],
        'rhat': tau_row['r_hat']
    }
    
    print(f"\nObservation error (τ):")
    print(f"  {tau_estimates['mean']:.4f} ± {tau_estimates['std']:.4f} "
          f"[{tau_estimates['hdi_low']:.4f}, {tau_estimates['hdi_high']:.4f}] "
          f"(R̂={tau_estimates['rhat']:.3f})")
    
    params['tau'] = tau_estimates
    
    # Initial factor values
    print(f"\nInitial factor values (f₀):")
    f0_estimates = []
    for i in range(k):
        f0_row = summary.loc[f'f0[{i}]']
        f0_mean = f0_row['mean']
        f0_std = f0_row['sd']
        f0_rhat = f0_row['r_hat']
        
        print(f"  Factor {i+1}: {f0_mean:.3f} ± {f0_std:.3f} "
              f"(R̂={f0_rhat:.3f})")
        
        f0_estimates.append({
            'mean': f0_mean, 'std': f0_std, 'rhat': f0_rhat
        })
    
    params['f0'] = f0_estimates
    
    return params


def analyze_factor_loadings(trace, task_ids, k=3, top_n=5):
    """
    Analyze and interpret factor loadings
    
    Parameters:
    -----------
    trace : ArviZ InferenceData
        MCMC trace from sampling
    task_ids : list
        List of task IDs corresponding to rows in loadings matrix
    k : int
        Number of factors
    top_n : int
        Number of top tasks to show per factor
        
    Returns:
    --------
    dict : Dictionary with loading analysis
    """
    
    print("\n" + "="*80)
    print("FACTOR LOADINGS ANALYSIS")
    print("="*80)
    
    # Extract posterior means for loadings
    loadings_mean = trace.posterior['loadings_matrix'].mean(dim=['chain', 'draw']).values
    loadings_std = trace.posterior['loadings_matrix'].std(dim=['chain', 'draw']).values
    
    # Get summary for convergence info
    summary = az.summary(trace)
    loadings_summary = summary.loc[summary.index.str.contains('loadings_raw')]
    
    print(f"Factor loadings summary statistics:")
    print(f"  Mean loading: {loadings_mean.mean():.4f}")
    print(f"  Loading range: [{loadings_mean.min():.4f}, {loadings_mean.max():.4f}]")
    if len(loadings_summary) > 0:
        print(f"  Worst R̂: {loadings_summary['r_hat'].max():.3f}")
        print(f"  Min ESS: {loadings_summary['ess_bulk'].min():.0f}")
    
    # Factor interpretation
    factor_analysis = {}
    
    print(f"\nTop {top_n} tasks per factor (by absolute loading):")
    for factor_idx in range(k):
        print(f"\n  === Factor {factor_idx + 1} ===")
        
        # Get loadings for this factor
        factor_loadings = loadings_mean[:, factor_idx]
        factor_loadings_std = loadings_std[:, factor_idx]
        
        # Sort by absolute loading value
        sorted_indices = np.argsort(np.abs(factor_loadings))[::-1]
        
        top_tasks = []
        for rank in range(min(top_n, len(task_ids))):
            task_idx = sorted_indices[rank]
            task_id = task_ids[task_idx]
            loading_val = factor_loadings[task_idx]
            loading_unc = factor_loadings_std[task_idx]
            
            print(f"    {rank+1}. Task {task_id}: {loading_val:.3f} ± {loading_unc:.3f}")
            
            top_tasks.append({
                'task_id': task_id,
                'task_idx': task_idx,
                'loading_mean': loading_val,
                'loading_std': loading_unc,
                'rank': rank + 1
            })
        
        factor_analysis[f'factor_{factor_idx + 1}'] = {
            'top_tasks': top_tasks,
            'mean_loading': float(factor_loadings.mean()),
            'std_loading': float(factor_loadings.std())
        }
    
    return factor_analysis, loadings_mean, loadings_std


def analyze_factor_evolution(trace, time_points, k=3):
    """
    Analyze factor evolution over time
    
    Parameters:
    -----------
    trace : ArviZ InferenceData
        MCMC trace from sampling
    time_points : array
        Time points (days from start)
    k : int
        Number of factors
        
    Returns:
    --------
    dict : Dictionary with factor evolution
    """
    
    print("\n" + "="*80)
    print("FACTOR EVOLUTION OVER TIME")
    print("="*80)
    
    # Extract factor values
    factors_mean = trace.posterior['factors'].mean(dim=['chain', 'draw']).values
    factors_std = trace.posterior['factors'].std(dim=['chain', 'draw']).values
    
    print("Factor values at key time points:")
    print(f"{'Time (days)':<12} ", end="")
    for i in range(k):
        print(f"Factor {i+1:<8}", end="  ")
    print()
    print("-" * (12 + k * 10))
    
    evolution_data = {}
    for t_idx, t in enumerate(time_points):
        print(f"{t:<12.0f} ", end="")
        
        factor_values_at_t = {}
        for factor_idx in range(k):
            factor_val = factors_mean[t_idx, factor_idx]
            factor_unc = factors_std[t_idx, factor_idx]
            print(f"{factor_val:>6.2f}±{factor_unc:.2f} ", end="")
            
            factor_values_at_t[f'factor_{factor_idx + 1}'] = {
                'mean': float(factor_val),
                'std': float(factor_unc)
            }
        
        evolution_data[f'time_{t}'] = factor_values_at_t
        print()
    
    # Calculate total improvement per factor
    print(f"\nTotal factor improvement (first to last time point):")
    for factor_idx in range(k):
        initial_val = factors_mean[0, factor_idx]
        final_val = factors_mean[-1, factor_idx]
        total_improvement = final_val - initial_val
        
        print(f"  Factor {factor_idx + 1}: {initial_val:.3f} → {final_val:.3f} "
              f"(Δ = {total_improvement:+.3f})")
    
    return evolution_data, factors_mean, factors_std


#%%
df = pd.read_csv('../../results/tables/df_model_test_scores.csv')
model_full, tp_full, y_obs_full, tasks_full = quick_test_with_data(
    df, k=3, max_tasks=None,  # Use ALL 99 tasks
    use_halfnormal=True,
    halfnormal_sigma=1.0
)

# 2. Fit the model (this will take time)
trace_full = fit_factor_model(
    model_full,
    draws=1000,        # Posterior samples per chain
    tune=2000,         # Tuning samples
    chains=4,          # Number of chains  
    target_accept=0.95 # High acceptance for better convergence
)


#%%

diagnostics = analyze_trace_diagnostics(trace_full)

# Analyze key parameters (mu, sigma, tau, f0)
params = analyze_factor_parameters(trace_full, k=3)

# Interpret factor loadings 
loadings_analysis, loadings_mean, loadings_std = analyze_factor_loadings(
    trace_full, tasks_full, k=3, top_n=5
)

# Analyze factor evolution over time
evolution_data, factors_mean, factors_std = analyze_factor_evolution(
    trace_full, tp_full, k=3
)