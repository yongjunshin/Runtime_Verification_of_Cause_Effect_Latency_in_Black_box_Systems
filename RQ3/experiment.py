#!/usr/bin/env python3
"""
RQ3 Experiment: Normal Distribution Time Series Generation

This module generates N samples from a normal distribution for RQ3 time series analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import argparse
import sys
import os

import os

# Add parent directory to path to access util modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)
from util.cause_effect_latency_verifier import verify_cause_effect_latency_new, VerificationResult


def generate_normal_time_series(n_samples: int, mean: float, std: float, 
                               seed: int = None) -> np.ndarray:
    """
    Generate N samples from a normal distribution (time series data)
    
    Args:
        n_samples: Number of samples to generate
        mean: Mean of the normal distribution
        std: Standard deviation of the normal distribution
        seed: Random seed for reproducibility (optional)
    
    Returns:
        Numpy array containing the time series samples
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate N samples from normal distribution
    time_series = np.random.normal(mean, std, n_samples)
    
    return time_series


def analyze_time_series(time_series: np.ndarray) -> dict:
    """
    Analyze the generated time series
    
    Args:
        time_series: Numpy array containing the time series data
    
    Returns:
        Dictionary containing analysis results
    """
    analysis = {
        'n_samples': len(time_series),
        'mean': np.mean(time_series),
        'std': np.std(time_series),
        'min': np.min(time_series),
        'max': np.max(time_series),
        'first_10': time_series[:10].tolist(),
        'last_10': time_series[-10:].tolist()
    }
    
    return analysis


def print_analysis(analysis: dict):
    """
    Print analysis results in a formatted way
    
    Args:
        analysis: Dictionary containing analysis results
    """
    print("=" * 80)
    print("RQ3 NORMAL DISTRIBUTION TIME SERIES ANALYSIS")
    print("=" * 80)
    print(f"Number of samples: {analysis['n_samples']}")
    print(f"Sample mean: {analysis['mean']:.4f}")
    print(f"Sample std: {analysis['std']:.4f}")
    print(f"Min value: {analysis['min']:.4f}")
    print(f"Max value: {analysis['max']:.4f}")
    print()
    
    print("First 10 samples:")
    print("-" * 40)
    for i, value in enumerate(analysis['first_10']):
        print(f"Sample {i+1:2d}: {value:8.4f}")
    
    print()
    print("Last 10 samples:")
    print("-" * 40)
    for i, value in enumerate(analysis['last_10']):
        print(f"Sample {analysis['n_samples']-9+i:2d}: {value:8.4f}")


def perform_latency_verification(time_series: np.ndarray, threshold: float, 
                                content: float, confidence: float, 
                                min_samples: int = 3, max_samples: int = None) -> VerificationResult:
    """
    Perform cause-effect latency verification on the time series using the new algorithm
    
    Args:
        time_series: The generated time series data
        threshold: Cause-effect latency requirement threshold (delta)
        content: Content (coverage) parameter p in (0,1)
        confidence: Confidence level gamma in (0,1)
        min_samples: Minimum number of samples required
        max_samples: Optional maximum number of samples
    
    Returns:
        VerificationResult containing the verification outcome
    """
    # Convert numpy array to list for the verifier
    samples = time_series.tolist()
    
    # Perform verification using the new algorithm
    result = verify_cause_effect_latency_new(
        samples=samples,
        threshold=threshold,
        content=content,
        confidence=confidence,
        min_samples=min_samples,
        max_samples=max_samples
    )
    
    return result


# Cache for normal distribution quantiles to avoid repeated expensive calculations
_norm_quantile_cache = {}

def calculate_true_upper_tolerance_limit(true_mean: float, true_std: float, content: float) -> float:
    """
    Calculate the p% upper tolerance limit of the true population
    
    Args:
        true_mean: True mean of the population
        true_std: True standard deviation of the population
        content: Content (coverage) parameter p in (0,1)
    
    Returns:
        The p% upper tolerance limit of the true population
    """
    from scipy import stats
    
    # For normal distribution, the p% upper tolerance limit is:
    # mean + z_p * std, where z_p is the p-th quantile of standard normal
    
    # Use cache to avoid repeated expensive normal distribution calculations
    if content not in _norm_quantile_cache:
        _norm_quantile_cache[content] = stats.norm.ppf(content)
    
    z_p = _norm_quantile_cache[content]
    upper_limit = true_mean + z_p * true_std
    
    return upper_limit


def print_verification_results(result: VerificationResult, threshold: float, 
                             content: float, confidence: float, min_samples: int, max_samples: int = None):
    """
    Print verification results in a formatted way
    
    Args:
        result: VerificationResult from the latency verifier
        threshold: Threshold used for verification
        content: Content (coverage) parameter used
        confidence: Confidence level used
        min_samples: Minimum samples used
        max_samples: Maximum samples used (if any)
    """
    print("=" * 80)
    print("CAUSE-EFFECT LATENCY VERIFICATION RESULTS")
    print("=" * 80)
    print(f"Verification parameters:")
    print(f"  Threshold (δ): {threshold}")
    print(f"  Content (p): {content}")
    print(f"  Confidence (γ): {confidence}")
    print(f"  Min samples: {min_samples}")
    if max_samples:
        print(f"  Max samples: {max_samples}")
    print()
    
    print(f"Verification verdict: {result.verdict.upper()}")
    print(f"Latency estimate: {result.mean_latency:.4f}")
    print(f"Number of used samples: {result.num_used_samples}")
    print(f"Number of passed samples: {result.num_passed_samples}")
    print()
    
    # Interpretation
    if result.verdict == 'pass':
        print("✓ VERIFICATION PASSED: Latency requirement is satisfied")
    elif result.verdict == 'fail':
        print("✗ VERIFICATION FAILED: Latency requirement is not satisfied")
    else:
        print("? VERIFICATION INCONCLUSIVE: No decision could be made")


def export_error_matrices_to_csv(false_positive_matrix, false_negative_matrix, latency_diff_matrix, CONTENT_CONFIGURATIONS, CONFIDENCE_CONFIGURATIONS):
    """Export both error rate matrices and latency difference matrices to CSV files"""
    import csv
    
    # Export False Positive Rate Matrix
    with open('false_positive_matrix.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header row
        header = ['content'] + [f'confidence_{conf:.2f}' for conf in CONFIDENCE_CONFIGURATIONS]
        writer.writerow(header)
        
        # Data rows
        for content in CONTENT_CONFIGURATIONS:
            row = [content] + [round(false_positive_matrix[content][conf], 3) for conf in CONFIDENCE_CONFIGURATIONS]
            writer.writerow(row)
    
    # Export False Negative Rate Matrix
    with open('false_negative_matrix.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header row
        header = ['content'] + [f'confidence_{conf:.2f}' for conf in CONFIDENCE_CONFIGURATIONS]
        writer.writerow(header)
        
        # Data rows
        for content in CONTENT_CONFIGURATIONS:
            row = [content] + [round(false_negative_matrix[content][conf], 3) for conf in CONFIDENCE_CONFIGURATIONS]
            writer.writerow(row)
    
    # Export Latency Difference Matrix
    with open('latency_diff_matrix.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header row
        header = ['content'] + [f'confidence_{conf:.2f}' for conf in CONFIDENCE_CONFIGURATIONS]
        writer.writerow(header)
        
        # Data rows
        for content in CONTENT_CONFIGURATIONS:
            row = [content] + [round(latency_diff_matrix[content][conf], 3) for conf in CONFIDENCE_CONFIGURATIONS]
            writer.writerow(row)
    
    
    print("CSV files exported:")
    print("  - false_positive_matrix.csv (False positive rates)")
    print("  - false_negative_matrix.csv (False negative rates)")
    print("  - latency_diff_matrix.csv (Latency differences vs true p% upper tolerance limit)")


def generate_heatmap(matrix, 
                    row_labels, 
                    col_labels, 
                    title: str = "Heatmap",
                    show_title: bool = True,
                    font_size: int = 12,
                    figure_width: float = 10.0,
                    figure_height: float = 8.0,
                    safe_thresholds: Optional[list] = None,
                    save_path: Optional[str] = None,
                    colormap: str = 'Reds'):
    """
    Generate a heatmap for a matrix with customizable features
    
    Args:
        matrix: 2D numpy array or list of lists containing the data
        row_labels: List of row labels (e.g., [(0.01, 0.01), (0.01, 0.05), ...])
        col_labels: List of column labels (e.g., [0.01, 0.02, 0.05, ...])
        title: Title for the heatmap
        show_title: Whether to display the title
        font_size: Global font size for all text elements
        figure_width: Width of the figure in inches
        figure_height: Height of the figure in inches
        safe_thresholds: List of thresholds for each row (if None, no safe coloring)
        save_path: Path to save the plot (if None, plot is not saved)
        colormap: Matplotlib colormap name ('Reds', 'Blues', etc.)
    """
    # Convert matrix to numpy array if it's not already
    matrix = np.array(matrix)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    
    # Create a custom approach for individual cell coloring
    if safe_thresholds is not None:
        # Create a color matrix where each cell gets its own color
        color_matrix = np.zeros((len(row_labels), len(col_labels), 4))  # RGBA
        
        # Get the base colormap
        cmap = plt.colormaps[colormap]
        vmin = np.min(matrix)
        vmax = np.max(matrix)
        
        # Normalize values for colormap
        if vmax > vmin:
            normalized_matrix = (matrix - vmin) / (vmax - vmin)
        else:
            normalized_matrix = np.zeros_like(matrix)
        
        # Color each cell individually
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                value = matrix[i, j]
                threshold = safe_thresholds[i] if i < len(safe_thresholds) else float('inf')
                
                if value < threshold:
                    # Light green for safe values
                    color_matrix[i, j] = [0.2, 0.8, 0.2, 1.0]  # Light green with alpha
                else:
                    # Use colormap for unsafe values
                    color_matrix[i, j] = cmap(normalized_matrix[i, j])
        
        # Display the color matrix
        ax.imshow(color_matrix, aspect='auto')
        
        # Create a custom colorbar for the red values only
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        
        # Create a colorbar that only shows the red scale
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.ax.tick_params(labelsize=font_size + 4)
        cbar.set_label('Error Rate (Red cells only)', fontsize=font_size + 4)
        
    else:
        # Use standard colormap
        im = ax.imshow(matrix, cmap=colormap, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=font_size + 4)
    
    # Set ticks and labels
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels([f'{x:.2f}' for x in col_labels], fontsize=font_size + 4)
    ax.set_yticklabels([f'{x:.2f}' for x in row_labels], fontsize=font_size + 4)
    
    # Set axis labels
    ax.set_xlabel('Confidence level γ', fontsize=font_size + 4)
    ax.set_ylabel('Content (coverage) parameter p', fontsize=font_size + 4)
    
    # Add title if requested
    if show_title:
        ax.set_title(title, fontsize=font_size + 6, fontweight='bold')
    
    # Add gray grid lines for cell borders
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
    
    # Add text annotations for each cell
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            value = matrix[i, j]
            
            # Check if this cell should be colored green (safe)
            is_safe = False
            if safe_thresholds is not None and i < len(safe_thresholds):
                threshold = safe_thresholds[i]
                is_safe = value < threshold
            
            # Choose text color based on whether it's safe or not
            if is_safe:
                text_color = 'white'  # White text on green background
            else:
                text_color = 'black' if value < np.mean(matrix) else 'white'
            
            # Format value display: round up to 3 decimal places
            
            formatted_value = f'{value:.3f}'
            
            ax.text(j, i, formatted_value, ha='center', va='center', 
                   color=text_color, fontsize=font_size + 2, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    
    return fig, ax


def generate_all_heatmaps(false_positive_matrix, false_negative_matrix, latency_diff_matrix, CONTENT_CONFIGURATIONS, CONFIDENCE_CONFIGURATIONS):
    """Generate heatmaps for all matrices"""
    
    # Convert matrices to numpy arrays and flip vertically so y-axis increases from bottom to top
    false_positive_data = np.flipud(np.array([[false_positive_matrix[content][confidence] for confidence in CONFIDENCE_CONFIGURATIONS] 
                                   for content in CONTENT_CONFIGURATIONS]))
    false_negative_data = np.flipud(np.array([[false_negative_matrix[content][confidence] for confidence in CONFIDENCE_CONFIGURATIONS] 
                                   for content in CONTENT_CONFIGURATIONS]))
    latency_diff_data = np.flipud(np.array([[latency_diff_matrix[content][confidence] for confidence in CONFIDENCE_CONFIGURATIONS] 
                                 for content in CONTENT_CONFIGURATIONS]))
    
    # Generate False Positive Rate Heatmap
    generate_heatmap(
        matrix=false_positive_data,
        row_labels=list(reversed(CONTENT_CONFIGURATIONS)),  # Reverse labels to match flipped matrix
        col_labels=CONFIDENCE_CONFIGURATIONS,
        title="False Positive Rate Matrix",
        show_title=False,
        font_size=14,
        figure_width=8,
        figure_height=6,
        safe_thresholds=None,  # No specific safe threshold for false positives
        save_path="false_positive_heatmap.png",
        colormap='Reds'
    )
    
    # Generate False Negative Rate Heatmap
    generate_heatmap(
        matrix=false_negative_data,
        row_labels=list(reversed(CONTENT_CONFIGURATIONS)),  # Reverse labels to match flipped matrix
        col_labels=CONFIDENCE_CONFIGURATIONS,
        title="False Negative Rate Matrix",
        show_title=False,
        font_size=14,
        figure_width=8,
        figure_height=6,
        safe_thresholds=None,  # No specific safe threshold for false negatives
        save_path="false_negative_heatmap.png",
        colormap='Reds'
    )
    
    # Generate Latency Difference Heatmap
    generate_heatmap(
        matrix=latency_diff_data,
        row_labels=list(reversed(CONTENT_CONFIGURATIONS)),  # Reverse labels to match flipped matrix
        col_labels=CONFIDENCE_CONFIGURATIONS,
        title="Latency Difference Matrix (vs true p% upper tolerance limit)",
        show_title=False,
        font_size=14,
        figure_width=8,
        figure_height=6,
        safe_thresholds=None,  # No safe threshold, simple white to red
        save_path="latency_diff_heatmap.png",
        colormap='Reds'
    )
    
    print("All heatmaps generated and saved!")


def generate_fixed_dataset(n_experiments: int = 1000, mean_range: tuple = (50, 150), 
                          std: float = 10.0, max_num_samples: int = 1000, seed: int = 333):
    """
    Generate a fixed dataset of true means and corresponding time series for fair evaluation
    
    Args:
        n_experiments: Number of experiments to generate
        mean_range: Tuple (min_mean, max_mean) for uniform distribution of true means
        std: Standard deviation of the distribution
        max_num_samples: Maximum number of samples per time series
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (true_means, time_series_list, true_upper_limits_dict)
        - true_means: Array of true means
        - time_series_list: List of time series arrays
        - true_upper_limits_dict: Dictionary mapping content values to true upper limits
    """
    # Set random seed for reproducible results
    np.random.seed(seed)
    
    # Generate uniform distribution of true means
    true_means = np.random.uniform(mean_range[0], mean_range[1], n_experiments)
    
    # Generate time series for each true mean
    time_series_list = []
    for i, test_mean in enumerate(true_means):
        time_series = generate_normal_time_series(max_num_samples, test_mean, std, seed=seed+i)
        time_series_list.append(time_series)
    
    # Calculate true upper tolerance limits for different content values
    # We'll calculate this for all possible content values to avoid recalculation
    content_values = [0.90, 0.92, 0.95, 0.97, 0.99]
    true_upper_limits_dict = {}
    
    for content in content_values:
        true_upper_limits_dict[content] = [calculate_true_upper_tolerance_limit(mean, std, content) for mean in true_means]
    
    return true_means, time_series_list, true_upper_limits_dict


def run_error_rate_test_with_fixed_data(true_means, time_series_list, true_upper_limits_dict,
                                       threshold: float = 100.0, content: float = 0.95,
                                       confidence: float = 0.95, max_num_samples: int = 1000):
    """
    Run multiple experiments to test false positive and false negative rates
    using pre-generated fixed dataset for fair evaluation
    
    Args:
        true_means: Array of true means (pre-generated)
        time_series_list: List of time series arrays (pre-generated)
        true_upper_limits_dict: Dictionary mapping content values to true upper limits
        threshold: Verification threshold
        content: Content (coverage) parameter p in (0,1)
        confidence: Confidence level gamma in (0,1)
        max_num_samples: Maximum number of samples to use
    
    Returns:
        Tuple of (false_positive_rate, false_negative_rate, latency_diff_mean, latency_diff_std)
    """
    n_experiments = len(true_means)
    
    # Get true upper limits for the specific content value
    true_upper_limits = true_upper_limits_dict[content]
    
    # Separate into False Positive and False Negative cases
    # False Positive: true upper limit >= threshold, but reported pass
    # False Negative: true upper limit < threshold, but reported fail
    negative_cases = np.array(true_upper_limits) >= threshold  # Should FAIL
    positive_cases = np.array(true_upper_limits) < threshold   # Should PASS
    
    false_positive_errors = 0
    false_positive_experiments = 0
    false_negative_errors = 0
    false_negative_experiments = 0
    
    # Track differences between estimated latency and true upper tolerance limit
    latency_diff_diffs = []
    
    # Run all experiments using the pre-generated data
    for i in range(n_experiments):
        true_upper_limit = true_upper_limits[i]
        time_series = time_series_list[i]
        
        result = perform_latency_verification(
            time_series=time_series,
            threshold=threshold,
            content=content,
            confidence=confidence,
            min_samples=3,
            max_samples=max_num_samples
        )
        
        # Calculate difference between estimated latency and true upper tolerance limit
        if not np.isnan(result.mean_latency):
            latency_diff = abs(result.mean_latency - true_upper_limit)
            latency_diff_diffs.append(latency_diff)
        
        if negative_cases[i]:  # Should FAIL (true upper limit >= threshold)
            false_positive_experiments += 1
            if result.verdict == 'pass':  # This is a false positive error
                false_positive_errors += 1
        else:  # Should PASS (true upper limit < threshold)
            false_negative_experiments += 1
            if result.verdict == 'fail':  # This is a false negative error
                false_negative_errors += 1
    
    false_positive_rate = false_positive_errors / false_positive_experiments if false_positive_experiments > 0 else 0
    false_negative_rate = false_negative_errors / false_negative_experiments if false_negative_experiments > 0 else 0
    
    # Calculate statistics for latency differences
    latency_diff_mean = np.mean(latency_diff_diffs) if latency_diff_diffs else 0
    latency_diff_std = np.std(latency_diff_diffs) if latency_diff_diffs else 0
    
    return false_positive_rate, false_negative_rate, latency_diff_mean, latency_diff_std


def run_error_rate_test(n_experiments: int = 1000, mean_range: tuple = (50, 150), 
                       threshold: float = 100.0, content: float = 0.95,
                       confidence: float = 0.95, std: float = 10.0, seed: int = 333):
    """
    Run multiple experiments to test false positive and false negative rates
    with uniformly distributed true means across a range (legacy function)
    
    Args:
        n_experiments: Number of experiments to run
        mean_range: Tuple (min_mean, max_mean) for uniform distribution of true means
        threshold: Verification threshold
        content: Content (coverage) parameter p in (0,1)
        confidence: Confidence level gamma in (0,1)
        std: Standard deviation of the distribution
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (false_positive_rate, false_negative_rate, latency_diff_mean, latency_diff_std)
    """
    # print("=" * 80)
    # print("ERROR RATE VERIFICATION TEST - UNIFORM MEAN DISTRIBUTION")
    # print("=" * 80)
    # print(f"Running {n_experiments} experiments...")
    # print(f"True mean range: {mean_range[0]} to {mean_range[1]}")
    # print(f"Threshold: {threshold}")
    # print(f"Target α: {alpha}, Target β: {beta}")
    # print()
    
    # Set random seed for reproducible results
    np.random.seed(seed)
    
    # Generate uniform distribution of true means
    true_means = np.random.uniform(mean_range[0], mean_range[1], n_experiments)
    
    # Calculate true upper tolerance limits for each true mean
    true_upper_limits = [calculate_true_upper_tolerance_limit(mean, std, content) for mean in true_means]
    
    # Separate into False Positive and False Negative cases
    # False Positive: true upper limit >= threshold, but reported pass
    # False Negative: true upper limit < threshold, but reported fail
    negative_cases = np.array(true_upper_limits) >= threshold  # Should FAIL
    positive_cases = np.array(true_upper_limits) < threshold   # Should PASS
    
    false_positive_errors = 0
    false_positive_experiments = 0
    false_negative_errors = 0
    false_negative_experiments = 0
    
    # Track differences between estimated latency and true upper tolerance limit
    latency_diff_diffs = []
    
    
    # print(f"Type I cases (true_mean >= {threshold}): {np.sum(type1_cases)} experiments")
    # print(f"Type II cases (true_mean < {threshold}): {np.sum(type2_cases)} experiments")
    # print()
    
    max_num_samples = 1000
    # Run all experiments
    for i in range(n_experiments):
        test_mean = true_means[i]
        true_upper_limit = true_upper_limits[i]
        time_series = generate_normal_time_series(max_num_samples, test_mean, std, seed=seed+i)
        
        result = perform_latency_verification(
            time_series=time_series,
            threshold=threshold,
            content=content,
            confidence=confidence,
            min_samples=3,
            max_samples=max_num_samples
        )
        
        # Calculate difference between estimated latency and true upper tolerance limit
        if not np.isnan(result.mean_latency):
            latency_diff = abs(result.mean_latency - true_upper_limit)
            latency_diff_diffs.append(latency_diff)
        
        if negative_cases[i]:  # Should FAIL (true upper limit >= threshold)
            false_positive_experiments += 1
            if result.verdict == 'pass':  # This is a false positive error
                false_positive_errors += 1
                # print(f"False positive case: verdict {result.verdict}, estimate {result.mean_latency:.4f}, true upper limit {true_upper_limit:.4f}")
        else:  # Should PASS (true upper limit < threshold)
            false_negative_experiments += 1
            if result.verdict == 'fail':  # This is a false negative error
                false_negative_errors += 1
                # print(f"False negative case: verdict {result.verdict}, estimate {result.mean_latency:.4f}, true upper limit {true_upper_limit:.4f}")
    
    false_positive_rate = false_positive_errors / false_positive_experiments if false_positive_experiments > 0 else 0
    # print(f"False positive rate: {false_positive_rate:.4f}, false positive errors: {false_positive_errors}/{false_positive_experiments}")
    false_negative_rate = false_negative_errors / false_negative_experiments if false_negative_experiments > 0 else 0
    # print(f"False negative rate: {false_negative_rate:.4f}, false negative errors: {false_negative_errors}/{false_negative_experiments}")
    
    # Calculate statistics for latency differences
    latency_diff_mean = np.mean(latency_diff_diffs) if latency_diff_diffs else 0
    latency_diff_std = np.std(latency_diff_diffs) if latency_diff_diffs else 0
    
    
    # print(f"Type I normalized mean diff: mean={type1_normalized_mean_diff_mean:.4f}, std={type1_normalized_mean_diff_std:.4f}")
    # print(f"Type II normalized mean diff: mean={type2_normalized_mean_diff_mean:.4f}, std={type2_normalized_mean_diff_std:.4f}")
    
    # Assessment
    # alpha_ok = false_positive_rate <= alpha  # Actual error rate should be ≤ target
    # beta_ok = false_negative_rate <= beta    # Actual error rate should be ≤ target

    # # Print results
    # print("=" * 80)
    # print("ERROR RATE TEST RESULTS")
    # print("=" * 80)
    # print(f"Type I Error Rate (α):")
    # print(f"  Target: {alpha:.3f}")
    # print(f"  Actual: {type1_error_rate:.3f}")
    # print(f"  Errors: {type1_errors}/{type1_experiments}")
    # print(f"  Difference: {abs(type1_error_rate - alpha):.3f}")
    # print()
    
    # print(f"Type II Error Rate (β):")
    # print(f"  Target: {beta:.3f}")
    # print(f"  Actual: {type2_error_rate:.3f}")
    # print(f"  Errors: {type2_errors}/{type2_experiments}")
    # print(f"  Difference: {abs(type2_error_rate - beta):.3f}")
    # print()
    
    # # Additional analysis
    # print("Additional Analysis:")
    # print(f"  Mean range: {mean_range[0]} to {mean_range[1]}")
    # print(f"  Threshold: {threshold}")
    # print(f"  H₁ mean (threshold × (1-γ)): {threshold * (1 - sensitivity_ratio):.1f}")
    # print(f"  Cases near threshold (±5): {np.sum(np.abs(true_means - threshold) <= 5)}")
    # print(f"  Cases far from threshold: {np.sum(np.abs(true_means - threshold) > 20)}")
    # print()
    
    # print("Assessment:")
    # if alpha_ok:
    #     print("✓ Type I error rate meets target (≤ {:.1%})".format(alpha))
    # else:
    #     print("✗ Type I error rate exceeds target ({:.1%} > {:.1%})".format(type1_error_rate, alpha))
    
    # if beta_ok:
    #     print("✓ Type II error rate meets target (≤ {:.1%})".format(beta))
    # else:
    #     print("✗ Type II error rate exceeds target ({:.1%} > {:.1%})".format(type2_error_rate, beta))
    
    # print("=" * 80)

    return false_positive_rate, false_negative_rate, latency_diff_mean, latency_diff_std


def main():
    """
    Main function for RQ3 experiment
    """
    # Time series generation parameters
    N_SAMPLES = 10000    # number of samples for error rate calculation of a (alpha, beta, sensitivity_ratio) configuration
    TRUE_MEAN_MIN = 90.0
    TRUE_MEAN_MAX = 110.0
    STD = 1
    SEED = 333
    
    # Verification parameters
    THRESHOLD = 100  # delta

    CONTENT_CONFIGURATIONS = [0.90, 0.92, 0.95, 0.97, 0.99]  # Content (coverage) parameter p
    CONFIDENCE_CONFIGURATIONS = [0.90, 0.92, 0.95, 0.97, 0.99]  # Confidence level gamma
    
    # Initialize result matrices
    false_positive_matrix = {}
    false_negative_matrix = {}
    latency_diff_matrix = {}
    
    print("Running comprehensive false positive/negative rate analysis...")
    print(f"Testing {len(CONTENT_CONFIGURATIONS)} content values × {len(CONFIDENCE_CONFIGURATIONS)} confidence values = {len(CONTENT_CONFIGURATIONS) * len(CONFIDENCE_CONFIGURATIONS)} total experiments")
    print()
    
    # Generate fixed dataset once for fair evaluation across all configurations
    print("Generating fixed dataset for fair evaluation...")
    true_means, time_series_list, true_upper_limits_dict = generate_fixed_dataset(
        n_experiments=N_SAMPLES,
        mean_range=(TRUE_MEAN_MIN, TRUE_MEAN_MAX),
        std=STD,
        max_num_samples=1000,
        seed=SEED
    )
    print(f"Generated {len(true_means)} experiments with fixed true means and time series")
    print()
    
    # Run all experiments using the same fixed dataset
    for i, CONTENT in enumerate(CONTENT_CONFIGURATIONS):
        print(f"Progress: {i+1}/{len(CONTENT_CONFIGURATIONS)} (content={CONTENT})")
        
        false_positive_matrix[CONTENT] = {}
        false_negative_matrix[CONTENT] = {}
        latency_diff_matrix[CONTENT] = {}
        
        for j, CONFIDENCE in enumerate(CONFIDENCE_CONFIGURATIONS):
            # Test using the same fixed dataset for fair comparison
            false_positive_rate, false_negative_rate, latency_diff_mean, latency_diff_std = run_error_rate_test_with_fixed_data(
                true_means=true_means,
                time_series_list=time_series_list,
                true_upper_limits_dict=true_upper_limits_dict,
                threshold=THRESHOLD,
                content=CONTENT,
                confidence=CONFIDENCE,
                max_num_samples=1000
            )
            
            false_positive_matrix[CONTENT][CONFIDENCE] = false_positive_rate
            false_negative_matrix[CONTENT][CONFIDENCE] = false_negative_rate
            latency_diff_matrix[CONTENT][CONFIDENCE] = latency_diff_mean
    
    print("\n" + "=" * 80)
    print("FALSE POSITIVE/NEGATIVE RATE ANALYSIS COMPLETED")
    print("=" * 80)
    
    # Print False Positive Rate Matrix
    print("\n1. FALSE POSITIVE RATE MATRIX")
    print("=" * 80)
    print("Content |", end="")
    for confidence in CONFIDENCE_CONFIGURATIONS:
        print(f" {confidence:6.2f}", end="")
    print()
    print("-" * (8 + 9 * len(CONFIDENCE_CONFIGURATIONS)))
    
    for content in CONTENT_CONFIGURATIONS:
        print(f"{content:7.2f} |", end="")
        for confidence in CONFIDENCE_CONFIGURATIONS:
            rate = false_positive_matrix[content][confidence]
            print(f" {rate:6.3f}", end="")
        print()
    
    # Print False Negative Rate Matrix
    print("\n2. FALSE NEGATIVE RATE MATRIX")
    print("=" * 80)
    print("Content |", end="")
    for confidence in CONFIDENCE_CONFIGURATIONS:
        print(f" {confidence:6.2f}", end="")
    print()
    print("-" * (8 + 9 * len(CONFIDENCE_CONFIGURATIONS)))
    
    for content in CONTENT_CONFIGURATIONS:
        print(f"{content:7.2f} |", end="")
        for confidence in CONFIDENCE_CONFIGURATIONS:
            rate = false_negative_matrix[content][confidence]
            print(f" {rate:6.3f}", end="")
        print()
    
    # Print Latency Difference Matrix
    print("\n3. LATENCY DIFFERENCE MATRIX (vs true p% upper tolerance limit)")
    print("=" * 80)
    print("Content |", end="")
    for confidence in CONFIDENCE_CONFIGURATIONS:
        print(f" {confidence:6.2f}", end="")
    print()
    print("-" * (8 + 9 * len(CONFIDENCE_CONFIGURATIONS)))
    
    for content in CONTENT_CONFIGURATIONS:
        print(f"{content:7.2f} |", end="")
        for confidence in CONFIDENCE_CONFIGURATIONS:
            diff = latency_diff_matrix[content][confidence]
            print(f" {diff:6.3f}", end="")
        print()
    
    
    # Export matrices to CSV files
    print("\n" + "=" * 80)
    print("EXPORTING MATRICES TO CSV")
    print("=" * 80)
    export_error_matrices_to_csv(false_positive_matrix, false_negative_matrix, latency_diff_matrix, CONTENT_CONFIGURATIONS, CONFIDENCE_CONFIGURATIONS)
    
    # Generate heatmaps
    print("\n" + "=" * 80)
    print("GENERATING HEATMAPS")
    print("=" * 80)
    generate_all_heatmaps(false_positive_matrix, false_negative_matrix, latency_diff_matrix, CONTENT_CONFIGURATIONS, CONFIDENCE_CONFIGURATIONS)
    print("\n" + "=" * 80)
    


if __name__ == "__main__":
    main()
