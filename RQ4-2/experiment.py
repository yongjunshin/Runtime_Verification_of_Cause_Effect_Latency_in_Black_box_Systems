#!/usr/bin/env python3
"""
RQ4-2-new Experiment: Verification Time Analysis

This script measures the verification time for cause-effect latency verification
across different iteration counts (10, 20, 30, ..., 100) comparing two versions:
- Version 1: Without caching
- Version 2: With caching
"""

import sys
import time
import csv
import random
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Literal
import statistics
import matplotlib.pyplot as plt
import numpy as np
import gc
import os
from dataclasses import dataclass
from scipy import stats
from scipy.stats import norm, chi2


@dataclass
class VerificationResult:
    """Result of the cause-effect latency verification"""
    verdict: Literal['pass', 'fail', 'none']
    mean_latency: float
    num_used_samples: int
    num_passed_samples: int


def welford_update(n: int, x_bar: float, M2: float, x: float) -> Tuple[int, float, float]:
    """
    Welford's online algorithm for computing mean and variance
    
    Args:
        n: Current number of samples
        x_bar: Current mean
        M2: Current sum of squared differences
        x: New sample value
    
    Returns:
        Tuple of (updated_n, updated_mean, updated_M2)
    """
    n += 1
    delta = x - x_bar
    x_bar += delta / n
    delta2 = x - x_bar
    M2 += delta * delta2
    
    return n, x_bar, M2


# Version 1: Without caching
def howe_k_factor_no_cache(u: float, confidence: float, nu: int, n: int) -> float:
    """
    Compute Howe's k-factor for normal tolerance intervals (without caching).
    
    Parameters
    ----------
    u : float
        Content proportion (coverage), e.g. 0.95 for 95%.
    confidence : float
        Confidence level, e.g. 0.99 for 99%.
    nu : int
        Degrees of freedom (usually n - 1).
    n : int
        Sample size.
        
    Returns
    -------
    k : float
        Howe's k factor.
    """
    # Input validation
    if not (0 < u < 1):
        raise ValueError("u (content) must be in (0,1)")
    if not (0 < confidence < 1):
        raise ValueError("gamma (confidence) must be in (0,1)")
    if nu <= 0:
        raise ValueError("nu (degrees of freedom) must be positive")
    if n <= 0:
        raise ValueError("n (sample size) must be positive")

    alpha = 1 - confidence
    
    # Calculate without caching - always compute fresh
    # Standard normal quantile for content u
    z_u = norm.ppf((1 + u) / 2.0)   # two-sided case
    
    # Chi-square critical value at gamma
    chi2_gamma = chi2.ppf(alpha, nu)
    
    # Howe's approximation
    k = math.sqrt((nu * (1 + 1/n) * z_u**2) / chi2_gamma)
    return k


# Version 2: With caching
_t_quantile_cache = {}

def clear_t_quantile_cache():
    """Clear the t-distribution quantile cache"""
    global _t_quantile_cache
    _t_quantile_cache.clear()

def howe_k_factor_with_cache(u: float, confidence: float, nu: int, n: int) -> float:
    """
    Compute Howe's k-factor for normal tolerance intervals (with caching).
    
    Parameters
    ----------
    u : float
        Content proportion (coverage), e.g. 0.95 for 95%.
    confidence : float
        Confidence level, e.g. 0.99 for 99%.
    nu : int
        Degrees of freedom (usually n - 1).
    n : int
        Sample size.
        
    Returns
    -------
    k : float
        Howe's k factor.
    """
    # Input validation
    if not (0 < u < 1):
        raise ValueError("u (content) must be in (0,1)")
    if not (0 < confidence < 1):
        raise ValueError("gamma (confidence) must be in (0,1)")
    if nu <= 0:
        raise ValueError("nu (degrees of freedom) must be positive")
    if n <= 0:
        raise ValueError("n (sample size) must be positive")

    alpha = 1 - confidence
    
    # Use cache to avoid repeated expensive statistical calculations
    cache_key = (u, alpha, nu)
    if cache_key not in _t_quantile_cache:
        # Standard normal quantile for content u
        z_u = norm.ppf((1 + u) / 2.0)   # two-sided case
        
        # Chi-square critical value at gamma
        chi2_gamma = chi2.ppf(alpha, nu)
        
        # Store both values in cache
        _t_quantile_cache[cache_key] = (z_u, chi2_gamma)
    
    z_u, chi2_gamma = _t_quantile_cache[cache_key]
    
    # Howe's approximation
    k = math.sqrt((nu * (1 + 1/n) * z_u**2) / chi2_gamma)
    return k


def verify_cause_effect_latency_no_cache(
    samples: List[float],
    threshold: float,
    content: float,
    confidence: float,
    min_samples: int = 3,
    max_samples: Optional[int] = None
) -> VerificationResult:
    """
    Verify cause-effect latency using tolerance intervals with Welford algorithm and Howe's k-factor (without caching)
    
    Args:
        samples: List of latency samples [l_hat_1, l_hat_2, ..., l_hat_k]
        threshold: Upper limit threshold (delta)
        content: Content (coverage) parameter p in (0,1)
        confidence: Confidence level gamma in (0,1)
        min_samples: Minimum number of samples required (n_min >= 3)
        max_samples: Optional maximum number of samples (n_max >= n_min)
    
    Returns:
        VerificationResult containing verdict, mean latency, and sample counts
    """
    # Input validation
    if not (0 < content < 1):
        raise ValueError("content must be in (0,1)")
    if not (0 < confidence < 1):
        raise ValueError("confidence must be in (0,1)")
    if min_samples < 3:
        raise ValueError("min_samples must be >= 3")
    if max_samples is not None and max_samples < min_samples:
        raise ValueError("max_samples must be >= min_samples")
    if not samples:
        raise ValueError("samples cannot be empty")
    
    # Initialize parameters (Init section)
    n = 0
    x_bar = 0.0
    M2 = 0.0
    psi = 'none'  # inconclusive -> none
    latency_estimate = float('nan')
    
    # Process each sample
    for x in samples:
        # Update statistics using Welford's algorithm
        n, x_bar, M2 = welford_update(n, x_bar, M2, x)
        
        # Check if we have enough samples
        if n < min_samples:
            continue
        
        # Calculate tolerance interval parameters
        nu = n - 1  # degrees of freedom
        s = np.sqrt(M2 / nu) if nu > 0 else 0.0
        
        # Calculate Howe's k-factor for two-sided tolerance interval (without cache)
        k = howe_k_factor_no_cache(content, confidence, nu, n)
        
        # Calculate upper and lower tolerance limits
        U = x_bar + k * s
        L = x_bar - k * s
        
        # Make decision based on tolerance limits
        if L <= threshold and U <= threshold:
            psi = 'pass'
            break
        elif L > threshold and U > threshold:
            psi = 'fail'
            break
        else:
            if n == max_samples:
                psi = 'fail'
                break
    
    # Calculate latency estimate if we have enough samples
    if n >= min_samples:
        nu = n - 1
        s = np.sqrt(M2 / nu) if nu > 0 else 0.0
        k = howe_k_factor_no_cache(content, confidence, nu, n)
        latency_estimate = x_bar + k * s
    
    return VerificationResult(
        verdict=psi,
        mean_latency=latency_estimate,
        num_used_samples=0,  # Not used in this algorithm
        num_passed_samples=0  # Not used in this algorithm
    )


def verify_cause_effect_latency_with_cache(
    samples: List[float],
    threshold: float,
    content: float,
    confidence: float,
    min_samples: int = 3,
    max_samples: Optional[int] = None
) -> VerificationResult:
    """
    Verify cause-effect latency using tolerance intervals with Welford algorithm and Howe's k-factor (with caching)
    
    Args:
        samples: List of latency samples [l_hat_1, l_hat_2, ..., l_hat_k]
        threshold: Upper limit threshold (delta)
        content: Content (coverage) parameter p in (0,1)
        confidence: Confidence level gamma in (0,1)
        min_samples: Minimum number of samples required (n_min >= 3)
        max_samples: Optional maximum number of samples (n_max >= n_min)
    
    Returns:
        VerificationResult containing verdict, mean latency, and sample counts
    """
    # Input validation
    if not (0 < content < 1):
        raise ValueError("content must be in (0,1)")
    if not (0 < confidence < 1):
        raise ValueError("confidence must be in (0,1)")
    if min_samples < 3:
        raise ValueError("min_samples must be >= 3")
    if max_samples is not None and max_samples < min_samples:
        raise ValueError("max_samples must be >= min_samples")
    if not samples:
        raise ValueError("samples cannot be empty")
    
    # Initialize parameters (Init section)
    n = 0
    x_bar = 0.0
    M2 = 0.0
    psi = 'none'  # inconclusive -> none
    latency_estimate = float('nan')
    
    # Process each sample
    for x in samples:
        # Update statistics using Welford's algorithm
        n, x_bar, M2 = welford_update(n, x_bar, M2, x)
        
        # Check if we have enough samples
        if n < min_samples:
            continue
        
        # Calculate tolerance interval parameters
        nu = n - 1  # degrees of freedom
        s = np.sqrt(M2 / nu) if nu > 0 else 0.0
        
        # Calculate Howe's k-factor for two-sided tolerance interval (with cache)
        k = howe_k_factor_with_cache(content, confidence, nu, n)
        
        # Calculate upper and lower tolerance limits
        U = x_bar + k * s
        L = x_bar - k * s
        
        # Make decision based on tolerance limits
        if L <= threshold and U <= threshold:
            psi = 'pass'
            break
        elif L > threshold and U > threshold:
            psi = 'fail'
            break
        else:
            if n == max_samples:
                psi = 'fail'
                break
    
    # Calculate latency estimate if we have enough samples
    if n >= min_samples:
        nu = n - 1
        s = np.sqrt(M2 / nu) if nu > 0 else 0.0
        k = howe_k_factor_with_cache(content, confidence, nu, n)
        latency_estimate = x_bar + k * s
    
    return VerificationResult(
        verdict=psi,
        mean_latency=latency_estimate,
        num_used_samples=0,  # Not used in this algorithm
        num_passed_samples=0  # Not used in this algorithm
    )


def generate_random_time_series(length: int, seed: int = None) -> List[float]:
    """
    Generate random time series data for verification testing
    
    Args:
        length: Length of the time series
        seed: Random seed for reproducibility
    
    Returns:
        List of random latency samples
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate samples with some realistic distribution
    # Mix of normal distributions with different means and variances
    samples = []
    
    # 60% samples from normal(80, 10) - mostly below threshold
    # 30% samples from normal(100, 15) - around threshold  
    # 10% samples from normal(120, 20) - above threshold
    
    n1 = int(0.6 * length)
    n2 = int(0.3 * length)
    n3 = length - n1 - n2
    
    samples.extend(np.random.normal(80, 10, n1).tolist())
    samples.extend(np.random.normal(100, 15, n2).tolist())
    samples.extend(np.random.normal(120, 20, n3).tolist())
    
    # Shuffle the samples
    random.shuffle(samples)
    
    return samples


def export_verification_time_results(results_data: List[Dict[str, Any]], output_file: str) -> None:
    """
    Export verification time results to CSV file
    
    Args:
        results_data: List of dictionaries containing verification time results
        output_file: Output CSV filename
    """
    # Create output directory if it doesn't exist
    output_dir = Path("verification_time_results")
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / output_file
    
    # Define CSV headers
    headers = ['iteration_count', 'version', 'run_number', 'verification_time']
    
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            
            # Write header
            writer.writeheader()
            
            # Write data rows
            for result in results_data:
                writer.writerow(result)

    except Exception as e:
        print(f"Error exporting verification time results: {e}")


def generate_verification_time_plot(verification_times_by_iteration: Dict[int, Dict[str, List[float]]], 
                                  output_file: str, show_title: bool = True, font_size: int = 12):
    """
    Generate a plot showing verification time distribution by iteration count for both versions
    Uses two subplots with shared x-axis - one for each version with optimized y-axis ranges
    
    Args:
        verification_times_by_iteration: Dictionary mapping iteration count to version results
        output_file: Output filename for the plot
        show_title: Whether to display the title
        font_size: Global font size for all text elements
    """
    # Create output directory if it doesn't exist
    output_dir = Path("verification_time_results")
    output_dir.mkdir(exist_ok=True)
    
    plot_path = output_dir / output_file
    
    # Prepare data for plotting
    iteration_counts = sorted(verification_times_by_iteration.keys())
    
    # Separate data for each version
    no_cache_times = []
    with_cache_times = []
    
    for count in iteration_counts:
        no_cache_times.append(verification_times_by_iteration[count]['no_cache'])
        with_cache_times.append(verification_times_by_iteration[count]['with_cache'])
    
    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Calculate optimal y-axis ranges for each version
    all_no_cache_times = [time for times_list in no_cache_times for time in times_list]
    all_with_cache_times = [time for times_list in with_cache_times for time in times_list]
    
    # Set y-axis limits with minimal padding for better data fit
    no_cache_min = max(0, min(all_no_cache_times) * 0.95)  # Reduced padding
    no_cache_max = max(all_no_cache_times) * 1.05  # Reduced padding
    with_cache_min = max(0, min(all_with_cache_times) * 0.95)  # Reduced padding
    with_cache_max = max(all_with_cache_times) * 1.05  # Reduced padding
    
    # Create box plots for no cache version (top subplot)
    box_plot_no_cache = ax1.boxplot(no_cache_times, 
                                   patch_artist=True,
                                   showfliers=True,
                                   labels=iteration_counts)
    
    # Create box plots for with cache version (bottom subplot)
    box_plot_with_cache = ax2.boxplot(with_cache_times, 
                                     patch_artist=True,
                                     showfliers=True,
                                     labels=iteration_counts)
    
    # Color the boxes
    for patch in box_plot_no_cache['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    for patch in box_plot_with_cache['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.7)
    
    # Customize the top subplot (no cache)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(no_cache_min, no_cache_max)
    ax1.tick_params(axis='x', labelsize=font_size)
    ax1.tick_params(axis='y', labelsize=font_size)
    
    # Add secondary y-axis for verifications per second (top)
    ax1_twin = ax1.twinx()
    ax1_twin.set_ylim(no_cache_min, no_cache_max)
    ax1_twin.set_yticks(ax1.get_yticks())
    
    # Create inverse tick labels for top subplot
    y1_ticks = ax1.get_yticks()
    y1_twin_labels = []
    for tick in y1_ticks:
        if tick > 0:
            ver_per_sec = 1000.0 / tick
            y1_twin_labels.append(f'{ver_per_sec:.0f}')
        else:
            y1_twin_labels.append('∞')
    
    ax1_twin.set_yticklabels(y1_twin_labels)
    ax1_twin.tick_params(axis='y', labelsize=font_size)
    
    # Customize the bottom subplot (with cache)
    ax2.set_xlabel('Number of Samples Used (n)', fontsize=font_size + 2)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(with_cache_min, with_cache_max)
    ax2.tick_params(axis='x', labelsize=font_size)
    ax2.tick_params(axis='y', labelsize=font_size)
    
    # Add secondary y-axis for verifications per second (bottom)
    ax2_twin = ax2.twinx()
    ax2_twin.set_ylim(with_cache_min, with_cache_max)
    ax2_twin.set_yticks(ax2.get_yticks())
    
    # Create inverse tick labels for bottom subplot
    y2_ticks = ax2.get_yticks()
    y2_twin_labels = []
    for tick in y2_ticks:
        if tick > 0:
            ver_per_sec = 1000.0 / tick
            y2_twin_labels.append(f'{ver_per_sec:.0f}')
        else:
            y2_twin_labels.append('∞')
    
    ax2_twin.set_yticklabels(y2_twin_labels)
    ax2_twin.tick_params(axis='y', labelsize=font_size)
    
    # Add single, center-aligned y-axis title for both subplots (left axis only)
    fig.text(0.02, 0.5, 'Verification Time (ms)', 
             rotation=90, va='center', ha='center', fontsize=font_size + 2)
    
    # Add single, center-aligned y-axis title for both subplots (right axis only)
    fig.text(0.98, 0.5, 'Verifications per Second', 
             rotation=90, va='center', ha='center', fontsize=font_size + 2)
    
    # Add overall title
    if show_title:
        fig.suptitle('Verification Time Distribution by Iteration Count', fontsize=font_size + 4, fontweight='bold')
    
    # Add separate legends for each subplot
    from matplotlib.patches import Patch
    
    # Legend for top subplot (without caching)
    legend_no_cache = [Patch(facecolor='lightblue', alpha=0.7, label='Without Caching')]
    ax1.legend(handles=legend_no_cache, loc='upper left', fontsize=font_size)
    
    # Legend for bottom subplot (with caching)
    legend_with_cache = [Patch(facecolor='lightcoral', alpha=0.7, label='With Caching')]
    ax2.legend(handles=legend_with_cache, loc='upper left', fontsize=font_size)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, left=0.14, right=0.83)  # Make room for suptitle and axis titles
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Verification time plot saved to: {plot_path}")
    
    # Show the plot
    plt.show()


def main():
    """Main experiment controller for RQ4-2-new"""
    print("RQ4-2-new: Verification Time Analysis")
    print("=" * 80)
    
    # ========================================
    # EXPERIMENT PARAMETER SETTINGS
    # ========================================
    
    # Iteration count range
    ITERATION_COUNTS = list(range(10, 101, 10))  # 10, 20, 30, ..., 100
    
    # Verification parameters
    THRESHOLD = 100.0
    CONTENT = 0.95  # 95% content
    CONFIDENCE = 0.95  # 95% confidence
    
    # Time series parameters
    TIME_SERIES_LENGTH = 10000
    NUM_RUNS_PER_ITERATION = 30  # Number of runs per iteration count per version
    BASE_SEED = 444
    
    # ========================================
    # PRINT EXPERIMENT CONFIGURATION
    # ========================================
    
    print("=== Experiment Configuration ===")
    print(f"Iteration counts: {ITERATION_COUNTS}")
    print(f"Time series length: {TIME_SERIES_LENGTH}")
    print(f"Runs per iteration per version: {NUM_RUNS_PER_ITERATION}")
    print(f"Effective data points per version: {NUM_RUNS_PER_ITERATION - 1} (excluding first run)")
    print(f"Threshold: {THRESHOLD}")
    print(f"Content: {CONTENT}")
    print(f"Confidence: {CONFIDENCE}")
    print(f"Base seed: {BASE_SEED}")
    print(f"Total experiments: {len(ITERATION_COUNTS) * 2 * (NUM_RUNS_PER_ITERATION - 1)}")
    
    # ========================================
    # EXPERIMENT EXECUTION
    # ========================================
    
    print(f"\nStarting verification time analysis...")
    experiment_start_time = time.time()
    
    # Initialize results collection
    all_verification_times = []
    verification_times_by_iteration = {count: {'no_cache': [], 'with_cache': []} for count in ITERATION_COUNTS}
    
    # Generate random time series datasets for each run in advance
    print(f"\nGenerating random time series datasets (length: {TIME_SERIES_LENGTH})...")
    time_series_datasets = []
    for run_num in range(1, NUM_RUNS_PER_ITERATION + 1):
        # Use different seed for each run to ensure different data
        run_seed = BASE_SEED + run_num
        time_series_data = generate_random_time_series(TIME_SERIES_LENGTH, run_seed)
        time_series_datasets.append(time_series_data)
    print(f"Generated {len(time_series_datasets)} time series datasets with {TIME_SERIES_LENGTH} samples each")
    
    # Warmup phase to eliminate cold start overhead
    print(f"\nPerforming warmup runs to eliminate cold start overhead...")
    warmup_data = time_series_datasets[0]  # Use first dataset for warmup
    warmup_iterations = [10, 50, 100]  # Test different iteration counts
    
    for warmup_iter in warmup_iterations:
        # Warmup both versions
        try:
            verify_cause_effect_latency_no_cache(
                samples=warmup_data,
                threshold=THRESHOLD,
                content=CONTENT,
                confidence=CONFIDENCE,
                min_samples=warmup_iter,
                max_samples=warmup_iter
            )
            verify_cause_effect_latency_with_cache(
                samples=warmup_data,
                threshold=THRESHOLD,
                content=CONTENT,
                confidence=CONFIDENCE,
                min_samples=warmup_iter,
                max_samples=warmup_iter
            )
        except Exception as e:
            print(f"    Warmup warning: {e}")
    
    # Additional targeted warmup for statistical functions
    print("Performing targeted statistical function warmup...")
    try:
        # Warmup norm.ppf and chi2.ppf functions specifically
        for _ in range(5):
            norm.ppf(0.95)
            chi2.ppf(0.05, 10)
            norm.ppf(0.975)
            chi2.ppf(0.025, 50)
    except Exception as e:
        print(f"    Statistical warmup warning: {e}")
    
    print("Warmup completed - cold start overhead eliminated")
    
    # Process each iteration count
    for iteration_count in ITERATION_COUNTS:
        print(f"\n{'='*60}")
        print(f"Processing Iteration Count: {iteration_count}")
        print(f"{'='*60}")
        
        # Clear cache before each iteration count to ensure fair comparison
        clear_t_quantile_cache()
        
        # Test both versions
        for version_name, version_func in [('no_cache', verify_cause_effect_latency_no_cache), 
                                         ('with_cache', verify_cause_effect_latency_with_cache)]:
            print(f"\n  Testing {version_name} version...")
            
            # Store timing results for this version to display all at once
            version_times = []
            
            # Run multiple times for this version
            for run_num in range(1, NUM_RUNS_PER_ITERATION + 1):
                try:
                    # Show progress
                    print(f"    Run {run_num:2d}/{NUM_RUNS_PER_ITERATION}", end="")
                    
                    # Get the pre-generated time series data for this run
                    time_series_data = time_series_datasets[run_num - 1]  # 0-indexed
                    
                    # Measure verification time with high precision
                    # Note: We use time.perf_counter() for more accurate timing measurements
                    start_time = time.perf_counter()
                    
                    # Run verification with fixed iteration count (min_samples = max_samples = iteration_count)
                    result = version_func(
                        samples=time_series_data,
                        threshold=THRESHOLD,
                        content=CONTENT,
                        confidence=CONFIDENCE,
                        min_samples=iteration_count,
                        max_samples=iteration_count
                    )
                    
                    end_time = time.perf_counter()
                    verification_time = (end_time - start_time) * 1000  # Convert to milliseconds
                    
                    # Skip the first run for all versions to avoid cache miss effects
                    if run_num > 1:
                        # Store results (only for runs 2 and onwards)
                        result_data = {
                            'iteration_count': iteration_count,
                            'version': version_name,
                            'run_number': run_num,
                            'verification_time': verification_time
                        }
                        
                        all_verification_times.append(result_data)
                        verification_times_by_iteration[iteration_count][version_name].append(verification_time)
                        version_times.append(verification_time)
                        
                        print(f" ✓ ({verification_time:.2f}ms) [included]")
                    else:
                        print(f" ✓ ({verification_time:.2f}ms) [excluded - first run]")
                    
                except Exception as e:
                    print(f" ✗ (Error: {e})")
                    continue
            
            # Display all timing results for this version
            if version_times:
                print(f"\n    {version_name} timing results (runs 2-{NUM_RUNS_PER_ITERATION}):")
                print(f"    " + "=" * 50)
                for i, time_ms in enumerate(version_times, 2):  # Start from run 2
                    print(f"    Run {i:2d}: {time_ms:8.2f}ms")
                print(f"    " + "=" * 50)
                mean_time = np.mean(version_times)
                std_time = np.std(version_times)
                min_time = np.min(version_times)
                max_time = np.max(version_times)
                print(f"    Mean: {mean_time:6.2f}ms, Std: {std_time:6.2f}ms, Min: {min_time:6.2f}ms, Max: {max_time:6.2f}ms")
                print(f"    Data points: {len(version_times)} (excluding first run)")
                print()
    
    # ========================================
    # RESULTS ANALYSIS AND VISUALIZATION
    # ========================================
    
    experiment_end_time = time.time()
    total_experiment_time = experiment_end_time - experiment_start_time
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETED")
    print(f"{'='*80}")
    print(f"Total experiment time: {total_experiment_time:.2f} seconds")
    print(f"Total verification experiments: {len(all_verification_times)}")
    
    # Print summary statistics
    print(f"\n=== Verification Time Summary ===")
    for iteration_count in ITERATION_COUNTS:
        print(f"\nIteration Count {iteration_count}:")
        for version_name in ['no_cache', 'with_cache']:
            if verification_times_by_iteration[iteration_count][version_name]:
                times = verification_times_by_iteration[iteration_count][version_name]
                mean_time = np.mean(times)
                std_time = np.std(times)
                min_time = np.min(times)
                max_time = np.max(times)
                print(f"  {version_name:12s}: Mean={mean_time:.2f}ms, Std={std_time:.2f}ms, "
                      f"Min={min_time:.2f}ms, Max={max_time:.2f}ms, N={len(times)}")
    
    # Export results to CSV
    print(f"\n=== Exporting Results ===")
    export_verification_time_results(all_verification_times, "verification_times.csv")
    
    # Generate plot
    print(f"\n=== Generating Visualization ===")
    generate_verification_time_plot(verification_times_by_iteration, "verification_time_plot.png", 
                                   show_title=False, font_size=24)
    
    print(f"\nResults saved in:")
    print(f"  - verification_time_results/verification_times.csv")
    print(f"  - verification_time_results/verification_time_plot.png")


if __name__ == "__main__":
    main()
