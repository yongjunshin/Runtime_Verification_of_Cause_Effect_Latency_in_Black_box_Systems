#!/usr/bin/env python3
"""
Total Analysis for Cause-Effect Chain Experiments

This script integrates all evaluation data with profile information to create
a comprehensive dataset for further analysis.
"""

import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def extract_profile_data(profile_file: Path) -> Dict[str, Any]:
    """
    Extract profile data based on the original comments
    
    Args:
        profile_file: Path to profile YAML file
        
    Returns:
        Dictionary containing extracted profile data
    """
    with open(profile_file, "r") as f:
        profile_data = yaml.safe_load(f)
    
    if 'tasks' not in profile_data:
        return {}
    
    tasks_data = profile_data['tasks']
    task_count = tasks_data.get('count', 0)
    
    # Extract task parameters (excluding 'count' key)
    task_params = {}
    for key, value in tasks_data.items():
        if key != 'count' and isinstance(value, dict):
            task_params[key] = value
    
    if not task_params:
        return {}
    
    # Calculate statistics as per original comments
    periods = [task['period'] for task in task_params.values()]
    wcets = [task['wcet'] for task in task_params.values()]
    utilizations = [task['wcet'] / task['period'] for task in task_params.values()]
    
    profile_info = {
        'task_count': task_count,
        'profile_repeat_index': int(profile_file.stem.split('_')[-1]),
        'min_period': min(periods),
        'mean_period': sum(periods) / len(periods),
        'max_period': max(periods),
        'min_wcet': min(wcets),
        'mean_wcet': sum(wcets) / len(wcets),
        'max_wcet': max(wcets),
        'min_wcet_period': min(utilizations),
        'mean_wcet_period': sum(utilizations) / len(utilizations),
        'max_wcet_period': max(utilizations),
        'minBoundOracleLength': 0,  # Always 0
        'maxBoundOracleLength': 2 * sum(periods),  # 2 * (sum of periods of all tasks)
        'minBoundEstimatedLength': 0,  # task1's period
        'maxBoundEstimatedLength': 3 * sum(periods),  # 3 * (sum of periods of all tasks)
        'sumPeriods': sum(periods)
    }
    
    return profile_info

def save_plots(fig, filename: str, output_dir: Path = None):
    """
    Save a matplotlib figure to a file
    
    Args:
        fig: Matplotlib figure object
        filename: Name of the file to save
        output_dir: Directory to save the plot (default: total_analysis_results)
    """
    if output_dir is None:
        output_dir = Path("total_analysis_results")
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    # Create full file path
    file_path = output_dir / filename
    
    # Save the plot
    try:
        fig.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved: {file_path}")
    except Exception as e:
        print(f"Error saving plot {filename}: {e}")
    
    # Close the figure to free memory
    plt.close(fig)


def draw_scatter_plot(total_df: pd.DataFrame, x_col: str, y_col: str, title: str, x_label: str, y_label: str, show_plot: bool = True, save_plot: bool = True, output_dir: Path = None, add_y_equals_x: bool = False, x_axis_range: tuple = None, y_axis_range: tuple = None, x_tick_labels: List[tuple] = None, y_tick_labels: List[tuple] = None, reference_lines: List[dict] = None, show_stats: bool = True, x_tick_postfix: str = None, y_tick_postfix: str = None, show_title: bool = True, fontsize: int = 14, legend_loc: str = None, legend_bbox_to_anchor: tuple = None):
    """
    Draw a scatter plot for the given columns
    
    Args:
        total_df: DataFrame containing the data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        show_plot: Whether to show the plot as popup
        save_plot: Whether to save the plot automatically
        output_dir: Directory to save the plot
        add_y_equals_x: Whether to add y=x line (perfect estimation reference)
        x_axis_range: Tuple (x_min, x_max) to set the x-axis range
        y_axis_range: Tuple (y_min, y_max) to set the y-axis range
        x_tick_labels: List of tuples (x_value, label) for custom x-axis tick labels
        y_tick_labels: List of tuples (y_value, label) for custom y-axis tick labels
        reference_lines: List of dictionaries with line specifications
                        Each dict can have: {'type': 'vertical'/'horizontal'/'diagonal', 
                                            'value': float, 'label': str, 'color': str, 'style': str}
        show_stats: Whether to show correlation and p-value statistics text
        x_tick_postfix: String to append to all x-axis tick labels (e.g., "ms", "%")
        y_tick_postfix: String to append to all y-axis tick labels (e.g., "ms", "%")
        show_title: Whether to show the plot title (default True)
        fontsize: Font size for axis labels (defaults to title font size)
        legend_loc: Legend location ('upper right', 'upper left', 'lower right', 'lower left', 'center', etc.)
        legend_bbox_to_anchor: Tuple (x, y) for custom legend position (e.g., (1.05, 1))
    """
    # Filter out NaN values
    valid_data = total_df.dropna(subset=[x_col, y_col])
    
    if valid_data.empty:
        print(f"No valid data for scatter plot: {x_col} vs {y_col}")
        return None
    
    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.scatter(valid_data[x_col], valid_data[y_col], alpha=0.6, s=15, facecolors='none', edgecolors='blue', linewidth=1)
    
    # Set axis ranges if provided
    if x_axis_range is not None:
        plt.xlim(x_axis_range)
    if y_axis_range is not None:
        plt.ylim(y_axis_range)
    
    # Add labels and title
    # Set axis labels with custom font size
    if fontsize is not None:
        plt.xlabel(x_label, fontsize=fontsize)
        plt.ylabel(y_label, fontsize=fontsize)
    else:
        plt.xlabel(x_label)
        plt.ylabel(y_label)
    
    # Set title if requested
    if show_title:
        if fontsize is not None:
            plt.title(title, fontsize=fontsize)
        else:
            plt.title(title)
    
    # Calculate correlation and p-value
    correlation, p_value = stats.pearsonr(valid_data[x_col], valid_data[y_col])
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Set custom tick labels if provided
    if x_tick_labels is not None:
        x_values = [x_val for x_val, _ in x_tick_labels]
        x_labels = [label for _, label in x_tick_labels]
        
        # Apply postfix to x-axis labels if provided
        if x_tick_postfix is not None:
            x_labels = [f"{label} {x_tick_postfix}" for label in x_labels]
        
        plt.xticks(x_values, x_labels, fontsize=fontsize)
    
    if y_tick_labels is not None:
        y_values = [y_val for y_val, _ in y_tick_labels]
        y_labels = [label for _, label in y_tick_labels]
        
        # Apply postfix to y-axis labels if provided
        if y_tick_postfix is not None:
            y_labels = [f"{label} {y_tick_postfix}" for label in y_labels]
        
        plt.yticks(y_values, y_labels, fontsize=fontsize)
    
    # Add reference lines if provided
    if reference_lines is not None:
        for line_spec in reference_lines:
            line_type = line_spec.get('type', 'vertical')
            value = line_spec.get('value', 0)
            label = line_spec.get('label', f'{line_type} line')
            color = line_spec.get('color', 'red')
            style = line_spec.get('style', '--')
            
            if line_type == 'vertical':
                plt.axvline(value, color=color, linestyle=style, linewidth=2, alpha=0.8, label=label)
            elif line_type == 'horizontal':
                plt.axhline(value, color=color, linestyle=style, linewidth=2, alpha=0.8, label=label)
            elif line_type == 'diagonal':
                # For diagonal lines, value should be a tuple (slope, intercept) or 'y=x'
                if value == 'y=x':
                    # Start y=x line from 0 for both axes
                    min_val = 0
                    max_val = max(valid_data[x_col].max(), valid_data[y_col].max())
                    plt.plot([min_val, max_val], [min_val, max_val], color=color, linestyle=style, 
                            linewidth=2, alpha=0.8, label=label)
                else:
                    # Assume value is (slope, intercept)
                    slope, intercept = value
                    x_vals = plt.xlim()
                    y_vals = [slope * x + intercept for x in x_vals]
                    plt.plot(x_vals, y_vals, color=color, linestyle=style, 
                            linewidth=2, alpha=0.8, label=label)
    
    # Add y=x line (perfect estimation line) if requested (legacy support)
    if add_y_equals_x:
        # Start y=x line from 0 for both axes
        min_val = 0
        max_val = max(valid_data[x_col].max(), valid_data[y_col].max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8, linewidth=2, label="y=x (Perfect estimation)")
    
    # Add legend if we have any lines
    if reference_lines is not None or add_y_equals_x:
        if legend_bbox_to_anchor is not None:
            plt.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor, fontsize=fontsize)
        else:
            plt.legend(loc=legend_loc, fontsize=fontsize)
    
    # Add correlation and p-value as text (if requested)
    if show_stats:
        p_value_text = f"r = {correlation:.3f}\np = {p_value:.3e}"
        plt.text(0.05, 0.95, p_value_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 fontsize=fontsize)
    
    plt.tight_layout()
    
    # Get the figure and save if requested
    fig = plt.gcf()
    if show_plot:
        plt.show()
    
    if save_plot:
        # Create filename from title
        filename = f"scatter_{x_col}_vs_{y_col}.png"
        filename = filename.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        save_plots(fig, filename, output_dir)
    
    return fig


def draw_histogram_plot(total_df: pd.DataFrame, col: str, title: str, x_label: str, y_label: str, show_plot: bool = True, save_plot: bool = True, output_dir: Path = None, vertical_lines: List[float] = None, vertical_line_labels: List[tuple] = None, x_tick_labels: List[tuple] = None, x_axis_range: tuple = None, mean_postfix: str = None, vertical_line_colors: List[str] = None, x_tick_postfix: str = None, show_title: bool = True, fontsize: int = 14, vertical_line_styles: List[dict] = None, legend_loc: str = None, legend_bbox_to_anchor: tuple = None):
    """
    Draw a histogram for the given column
    
    Args:
        total_df: DataFrame containing the data
        col: Column name to plot
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        show_plot: Whether to show the plot as popup
        save_plot: Whether to save the plot automatically
        output_dir: Directory to save the plot
        vertical_lines: List of x-axis values where vertical lines should be drawn (simple format)
        vertical_line_labels: List of tuples (x_value, label) for custom vertical line labels
        x_tick_labels: List of tuples (x_value, label) for custom x-axis tick labels
        x_axis_range: Tuple (x_min, x_max) to set the x-axis range
        mean_postfix: String to append to the mean legend (e.g., "ms", "%", "units")
        vertical_line_colors: List of colors for vertical lines (e.g., ["red", "blue", "green"])
        x_tick_postfix: String to append to all x-axis tick labels (e.g., "ms", "%")
        show_title: Whether to show the plot title (default True)
        fontsize: Font size for axis labels (defaults to title font size)
        vertical_line_styles: List of dictionaries with line style specifications
                            Each dict can have: {'value': float, 'label': str, 'color': str, 'style': str}
        legend_loc: Legend location ('upper right', 'upper left', 'lower right', 'lower left', 'center', etc.)
        legend_bbox_to_anchor: Tuple (x, y) for custom legend position (e.g., (1.05, 1))
    """
    # Filter out NaN values
    valid_data = total_df[col].dropna()
    
    if valid_data.empty:
        print(f"No valid data for histogram: {col}")
        return None
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    
    # Plot histogram
    plt.hist(valid_data, bins=30, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Set x-axis range if provided
    if x_axis_range is not None:
        plt.xlim(x_axis_range)
    
    # Add labels and title
    # Set axis labels with custom font size
    if fontsize is not None:
        plt.xlabel(x_label, fontsize=fontsize)
        plt.ylabel(y_label, fontsize=fontsize)
    else:
        plt.xlabel(x_label)
        plt.ylabel(y_label)
    
    # Set title if requested
    if show_title:
        if fontsize is not None:
            plt.title(title, fontsize=fontsize)
        else:
            plt.title(title)
        
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Set fontsize for y-axis ticks
    plt.yticks(fontsize=fontsize)
    
    # Set custom x-tick labels if provided
    if x_tick_labels is not None:
        # Extract x values and labels
        x_values = [x_val for x_val, _ in x_tick_labels]
        x_labels = [label for _, label in x_tick_labels]
        
        # Apply postfix to x-axis labels if provided
        if x_tick_postfix is not None:
            x_labels = [f"{label} {x_tick_postfix}" for label in x_labels]
        
        # Set the x-ticks and labels
        plt.xticks(x_values, x_labels, fontsize=fontsize)
    elif x_tick_postfix is not None:
        # Apply postfix to default tick labels
        ax = plt.gca()
        current_ticks = ax.get_xticks()
        current_labels = [str(tick) for tick in current_ticks]
        new_labels = [f"{label} {x_tick_postfix}" for label in current_labels]
        plt.xticks(current_ticks, new_labels, fontsize=fontsize)
    
    # Add statistics text
    mean_val = valid_data.mean()
    std_val = valid_data.std()
    
    # Create mean label with optional postfix
    mean_label = f'Mean: {mean_val:.2f}'
    if mean_postfix is not None:
        mean_label += f' {mean_postfix}'
    
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=mean_label)
    
    # Add vertical lines at specified x-axis values
    if vertical_line_styles is not None:
        # Use new style system with full control
        for line_style in vertical_line_styles:
            x_val = line_style.get('value', 0)
            label = line_style.get('label', f'Line: {x_val:.2f}')
            color = line_style.get('color', 'green')
            style = line_style.get('style', '--')
            
            plt.axvline(x_val, color=color, linestyle=style, linewidth=2, alpha=0.8, 
                       label=label)
    elif vertical_line_labels is not None:
        # Use custom labels from tuples (x_value, label)
        for i, (x_val, label) in enumerate(vertical_line_labels):
            # Get color for this line (cycle through colors if provided)
            if vertical_line_colors is not None and len(vertical_line_colors) > 0:
                color = vertical_line_colors[i % len(vertical_line_colors)]
            else:
                color = 'green'  # Default color
            
            plt.axvline(x_val, color=color, linestyle='--', linewidth=2, alpha=0.8, 
                       label=label)
    elif vertical_lines is not None:
        # Use default labels with x values
        for i, x_val in enumerate(vertical_lines):
            # Get color for this line (cycle through colors if provided)
            if vertical_line_colors is not None and len(vertical_line_colors) > 0:
                color = vertical_line_colors[i % len(vertical_line_colors)]
            else:
                color = 'green'  # Default color
            
            plt.axvline(x_val, color=color, linestyle='--', linewidth=2, alpha=0.8, 
                       label=f'Line {i+1}: {x_val:.2f}')
    
    if legend_bbox_to_anchor is not None:
        plt.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor, fontsize=fontsize)
    else:
        plt.legend(loc=legend_loc, fontsize=fontsize)
    
    # Add statistics text box
    stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nCount: {len(valid_data)}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=fontsize)
    
    plt.tight_layout()
    
    # Get the figure and save if requested
    fig = plt.gcf()
    if show_plot:
        plt.show()
    
    if save_plot:
        # Create filename from title
        filename = f"histogram_{col}.png"
        filename = filename.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        save_plots(fig, filename, output_dir)
    
    return fig

def perform_spearman_correlation_analysis(total_df: pd.DataFrame):
    """
    Perform Spearman correlation analysis for all factors with error_percentage
    
    Args:
        total_df: DataFrame containing the data
    """
    print("=" * 60)
    print("SPEARMAN CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Define all factors to analyze
    factors = ['task_count', 'min_period', 'mean_period', 'max_period', 
               'min_wcet', 'mean_wcet', 'max_wcet', 
               'min_wcet_period', 'mean_wcet_period', 'max_wcet_period']
    
    # Prepare data - remove NaN values
    analysis_df = total_df[factors + ['error_percentage']].dropna()
    
    if analysis_df.empty:
        print("No valid data for correlation analysis")
        return
    
    print(f"Sample size: {len(analysis_df)}")
    print()
    
    # Calculate Spearman correlations with error_percentage
    correlations = []
    p_values = []
    
    for factor in factors:
        if factor in analysis_df.columns:
            corr, p_val = stats.spearmanr(analysis_df[factor], analysis_df['error_percentage'])
            correlations.append(corr)
            p_values.append(p_val)
        else:
            correlations.append(np.nan)
            p_values.append(np.nan)
    
    # Create results table
    print("SPEARMAN CORRELATION RESULTS")
    print("-" * 80)
    print(f"{'Factor':<20} {'Correlation':<12} {'p-value':<12} {'Significance':<12} {'Strength':<12}")
    print("-" * 80)
    
    # Sort by absolute correlation strength
    factor_data = list(zip(factors, correlations, p_values))
    factor_data.sort(key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True)
    
    for factor, corr, p_val in factor_data:
        if np.isnan(corr):
            print(f"{factor:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
            continue
            
        # Determine significance
        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = ""
        
        # Determine correlation strength
        abs_corr = abs(corr)
        if abs_corr >= 0.7:
            strength = "Strong"
        elif abs_corr >= 0.5:
            strength = "Moderate"
        elif abs_corr >= 0.3:
            strength = "Weak"
        else:
            strength = "Very Weak"
        
        # Format p-value
        if p_val < 0.001:
            if p_val < 1e-10:
                p_val_str = "< 1e-10"
            else:
                p_val_str = f"{p_val:.2e}"
        else:
            p_val_str = f"{p_val:.4f}"
        
        print(f"{factor:<20} {corr:<12.4f} {p_val_str:<12} {sig:<12} {strength:<12}")
    
    print("-" * 80)
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05")
    print()
    
    # Correlation strength interpretation
    print("CORRELATION STRENGTH INTERPRETATION")
    print("-" * 50)
    print("|r| >= 0.7: Strong correlation")
    print("0.5 <= |r| < 0.7: Moderate correlation")
    print("0.3 <= |r| < 0.5: Weak correlation")
    print("|r| < 0.3: Very weak correlation")
    print()
    
    # Factor correlation matrix (between predictors)
    print("FACTOR CORRELATION MATRIX (Spearman)")
    print("-" * 60)
    factor_corr_matrix = analysis_df[factors].corr(method='spearman')
    print(factor_corr_matrix.round(4))
    print()
    
    # Multicollinearity warnings
    print("MULTICOLLINEARITY CHECK")
    print("-" * 40)
    high_corr_pairs = []
    for i in range(len(factors)):
        for j in range(i+1, len(factors)):
            if factors[i] in factor_corr_matrix.columns and factors[j] in factor_corr_matrix.columns:
                corr_val = abs(factor_corr_matrix.loc[factors[i], factors[j]])
                if corr_val > 0.8:
                    high_corr_pairs.append((factors[i], factors[j], corr_val))
    
    if high_corr_pairs:
        print("WARNING: High correlations detected (|r| > 0.8):")
        for factor1, factor2, corr_val in high_corr_pairs:
            print(f"  {factor1} <-> {factor2}: {corr_val:.4f}")
        print("Consider removing one of each highly correlated pair for regression.")
    else:
        print("No high correlations detected. All factors can be included in regression.")
    print()
    
    return {
        'correlations': dict(zip(factors, correlations)),
        'p_values': dict(zip(factors, p_values)),
        'factor_correlation_matrix': factor_corr_matrix,
        'high_correlation_pairs': high_corr_pairs
    }

def perform_multiple_regression_table(total_df: pd.DataFrame):
    """
    Perform multiple linear regression analysis and display results in table format
    
    Args:
        total_df: DataFrame containing the data
    """
    print("=" * 60)
    print("MULTIPLE LINEAR REGRESSION ANALYSIS")
    print("=" * 60)
    
    # Prepare data - remove NaN values
    analysis_df = total_df[['task_count', 'mean_period', 'mean_wcet_period', 'error_percentage']].dropna()
    
    if analysis_df.empty:
        print("No valid data for regression analysis")
        return
    
    print(f"Sample size: {len(analysis_df)}")
    print()
    
    # Prepare features and target
    X = analysis_df[['task_count', 'mean_period', 'mean_wcet_period']]
    y = analysis_df['error_percentage']
    
    # Standardize features for comparison
    X_mean = X.mean()
    X_std = X.std()
    X_scaled = (X - X_mean) / X_std
    
    # Manual multiple linear regression calculation
    # Add intercept column
    X_with_intercept = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    
    # Calculate coefficients using normal equation: (X'X)^(-1)X'y
    XtX = X_with_intercept.T @ X_with_intercept
    XtX_inv = np.linalg.inv(XtX)
    Xty = X_with_intercept.T @ y
    coefficients = XtX_inv @ Xty
    
    # Calculate R-squared
    y_pred = X_with_intercept @ coefficients
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Calculate adjusted R-squared
    n = len(y)
    k = X.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    
    # Calculate standard errors and t-statistics
    mse = ss_res / (n - k - 1)
    var_coef = mse * np.diag(XtX_inv)
    se_coef = np.sqrt(var_coef)
    t_stats = coefficients / se_coef
    
    # Calculate p-values
    from scipy.stats import t
    p_values = 2 * (1 - t.cdf(np.abs(t_stats), n - k - 1))
    
    # Handle extremely small p-values (set minimum to machine precision)
    min_p_value = 1e-16  # Machine precision limit
    p_values = np.maximum(p_values, min_p_value)
    
    # Create results table
    feature_names = ['Intercept', 'Task Count', 'Mean Period', 'Mean WCET/Period', 'Min Period', 'Max Period',
                     'Min WCET', 'Mean WCET', 'Max WCET', 'Min WCET/Period', 'Max WCET/Period']
    
    print("REGRESSION RESULTS TABLE")
    print("-" * 80)
    print(f"{'Variable':<18} {'Coefficient':<12} {'Std Error':<12} {'t-stat':<10} {'p-value':<12} {'Significance':<12}")
    print("-" * 80)
    
    for i, (name, coef, se, t_stat, p_val) in enumerate(zip(feature_names, coefficients, se_coef, t_stats, p_values)):
        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = ""
        
        # Format p-value in scientific notation
        if p_val < 0.001:
            if p_val < 1e-10:
                p_val_str = "< 1e-10"
            else:
                p_val_str = f"{p_val:.2e}"
        else:
            p_val_str = f"{p_val:.4f}"
        
        print(f"{name:<18} {coef:<12.4f} {se:<12.4f} {t_stat:<10.4f} {p_val_str:<12} {sig:<12}")
    
    print("-" * 80)
    print(f"R-squared: {r2:.4f}")
    print(f"Adjusted R-squared: {adj_r2:.4f}")
    print(f"Sample size: {n}")
    print()
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05")
    print()
    
    # Impact ranking (excluding intercept)
    print("IMPACT RANKING (by absolute standardized coefficient)")
    print("-" * 60)
    impact_data = []
    for i in range(1, len(coefficients)):  # Skip intercept
        name = feature_names[i]
        coef = coefficients[i]
        abs_coef = abs(coef)
        p_val = p_values[i]
        impact_data.append((name, coef, abs_coef, p_val))
    
    # Sort by absolute coefficient value
    impact_data.sort(key=lambda x: x[2], reverse=True)
    
    for i, (name, coef, abs_coef, p_val) in enumerate(impact_data, 1):
        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = ""
        
        # Format p-value in scientific notation for impact ranking
        if p_val < 0.001:
            if p_val < 1e-10:
                p_val_str = "< 1e-10"
            else:
                p_val_str = f"{p_val:.2e}"
        else:
            p_val_str = f"{p_val:.4f}"
        
        print(f"{i:2d}. {name:<18} {coef:>8.4f} (p={p_val_str}) {sig}")
    
    print()
    
    # Correlation matrix
    print("CORRELATION MATRIX")
    print("-" * 40)
    corr_matrix = analysis_df[['task_count', 'mean_period', 'mean_wcet_period', 
                               'error_percentage']].corr()
    print(corr_matrix.round(4))
    
    return {
        'coefficients': coefficients,
        'r2': r2,
        'adj_r2': adj_r2,
        'p_values': p_values,
        'correlation_matrix': corr_matrix
    }

def main():
    """Main function to generate integrated data file"""
    print("Total Analysis - Generating Integrated Data File")
    print("=" * 50)
    
    profile_dir = Path("random_profiles")
    eval_result_dir = Path("evaluation_results")
    total_analysis_dir = Path("total_analysis_results")
    total_eval_csv = total_analysis_dir / "total_evaluation_results.csv"
    graph_font_size = 14
    
    if not profile_dir.exists():
        print(f"Error: Profile directory {profile_dir} not found")
        return
    
    if not eval_result_dir.exists():
        print(f"Error: Evaluation results directory {eval_result_dir} not found")
        return
    
    # Create total analysis results directory if it doesn't exist
    total_analysis_dir.mkdir(exist_ok=True)
    
    all_results = []
    
    for profile_file in profile_dir.glob("*.yaml"):
        profile_name = profile_file.stem
        eval_file = eval_result_dir / f"evaluation_results_{profile_name}.yaml.csv"
        
        print(f"Processing {profile_name}...")
        
        if not eval_file.exists():
            print(f"  Warning: Evaluation results file not found for {profile_name}")
            continue
        
        # Extract profile data
        profile_info = extract_profile_data(profile_file)
        if not profile_info:
            print(f"  Warning: Could not extract data from {profile_name}")
            continue
        
        # Load evaluation results
        try:
            eval_df = pd.read_csv(eval_file)
            # change key name "repeat" to "simulation_repeat_index"
            eval_df = eval_df.rename(columns={'repeat': 'simulation_repeat_index'})
            
            # Add profile data to each evaluation row
            for key, value in profile_info.items():
                eval_df[key] = value
            
            # Reorder columns to put profile data first
            profile_cols = list(profile_info.keys())
            eval_cols = [col for col in eval_df.columns if col not in profile_cols]
            eval_df = eval_df[profile_cols + eval_cols]
            
            all_results.append(eval_df)
            print(f"  ✓ Added {len(eval_df)} rows")
            
        except Exception as e:
            print(f"  Error processing {profile_name}: {e}")
            continue
    
    if not all_results:
        print("No data to process. Exiting.")
        return
    
    # Combine all results
    print(f"\nCombining results...")
    total_df = pd.concat(all_results, ignore_index=True)
    
    # Sort by task_count, profile_repeat_index, simulation_repeat_index (increasing order)
    print("Sorting data...")
    total_df = total_df.sort_values(['task_count', 'profile_repeat_index', 'simulation_repeat_index'], ascending=True)

    # add columns 
    # 'normalized_oracle_chain_length': 'oracle_chain_length'/'maxBoundEstimatedLength'.
    # 'normalized_estimated_chain_length': 'estimated_chain_length'/'maxBoundEstimatedLength'.
    total_df['normalized_oracle_chain_length'] = total_df['oracle_chain_length'] / total_df['sumPeriods']
    total_df['normalized_estimated_chain_length'] = total_df['estimated_chain_length'] / total_df['sumPeriods']

    # Save integrated results
    print(f"Saving integrated data to {total_eval_csv}...")
    total_df.to_csv(total_eval_csv, index=False)
    
    print(f"\nComplete! Integrated data saved to: {total_eval_csv}")
    print(f"Total rows: {len(total_df)}")
    print(f"Total profiles processed: {len(all_results)}")


    print("=" * 25, 'Basic distributions of the data', "=" * 25)
    # print("Draw histogram plot (x: task_count)")
    # draw_histogram_plot(total_df, 'task_count', 
    #                     'Task Count Distribution', 'Task Count', 'Number of samples',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir)
    # print()

    # print("Draw histogram plot (x: min_period)")
    # draw_histogram_plot(total_df, 'min_period', 
    #                     'Min Period Distribution', 'Min Period', 'Number of samples',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir)
    # print()
    
    # print("Draw histogram plot (x: mean_period)")
    # draw_histogram_plot(total_df, 'mean_period', 
    #                     'Mean Period Distribution', 'Mean Period', 'Number of samples',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir)
    # print()
    
    # print("Draw histogram plot (x: max_period)")
    # draw_histogram_plot(total_df, 'max_period', 
    #                     'Max Period Distribution', 'Max Period', 'Number of samples',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir)
    # print()
    
    # print("Draw histogram plot (x: min_wcet)")
    # draw_histogram_plot(total_df, 'min_wcet', 
    #                     'Min WCET Distribution', 'Min WCET', 'Number of samples',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir)
    # print()

    # print("Draw histogram plot (x: mean_wcet)")
    # draw_histogram_plot(total_df, 'mean_wcet', 
    #                     'Mean WCET Distribution', 'Mean WCET', 'Number of samples',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir)
    # print()

    # print("Draw histogram plot (x: max_wcet)")
    # draw_histogram_plot(total_df, 'max_wcet', 
    #                     'Max WCET Distribution', 'Max WCET', 'Number of samples',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir)
    # print()


    # print("Draw histogram plot (x: min_wcet_period)")
    # draw_histogram_plot(total_df, 'min_wcet_period', 
    #                     'Min WCET/Period Distribution', 'Min WCET/Period', 'Number of samples',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir)
    # print()

    # print("Draw histogram plot (x: mean_wcet_period)")
    # draw_histogram_plot(total_df, 'mean_wcet_period', 
    #                     'Mean WCET/Period Distribution', 'Mean WCET/Period', 'Number of samples',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir)
    # print()

    # print("Draw histogram plot (x: max_wcet_period)")
    # draw_histogram_plot(total_df, 'max_wcet_period', 
    #                     'Max WCET/Period Distribution', 'Max WCET/Period', 'Number of samples',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir)
    # print()

    # print("Draw histogram plot (x: oracle_chain_length)")
    # draw_histogram_plot(total_df, 'oracle_chain_length', 
    #                     'Oracle Chain Length Distribution', 'Oracle Chain Length', 'Number of samples',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir)
    # print()
    

    # print("Draw histogram plot (x: estimated_chain_length)")
    # draw_histogram_plot(total_df, 'estimated_chain_length', 
    #                     'Estimated Chain Length Distribution', 'Estimated Chain Length', 'Number of samples',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir)
    # print()

    print("Draw histogram plot (x: normalized_oracle_chain_length)")
    draw_histogram_plot(total_df, 'normalized_oracle_chain_length', 
                        'Normalized Oracle Chain Length Distribution', r'Oracle chain length $l(\overleftarrow{ac})$ normalized by $\sum{T}$', 'Number of samples',
                      show_plot=True, save_plot=True, output_dir=total_analysis_dir, 
                      vertical_line_styles=[
                          {'value': 0.0, 'label': r'$l(\overleftarrow{ac})_{min}:0$', 'color': 'blue', 'style': ':'},
                          {'value': 2.0, 'label': r'$l(\overleftarrow{ac})_{max}:2\sum{T}$', 'color': 'blue', 'style': '-.'}
                      ],
                      x_tick_labels=[(0.0, r'$0$'), (1.0, r'$\sum{T}$'), (2.0, r'$2\sum{T}$'), (3.0, r'$3\sum{T}$')],
                      x_axis_range=(-0.1, 3.1), mean_postfix=r'$\sum{T}$',
                      show_title=False, fontsize=graph_font_size)
    print()
    
    
    print("Draw histogram plot (x: normalized_estimated_chain_length)")
    draw_histogram_plot(total_df, 'normalized_estimated_chain_length', 
                        'Normalized Estimated Chain Length Distribution', r'Estimated chain length $l(\widehat{\overleftarrow{ac}})$ normalized by $\sum{T}$', 'Number of samples',
                      show_plot=True, save_plot=True, output_dir=total_analysis_dir, 
                      vertical_line_styles=[
                          {'value': 0.0, 'label': r'$l(\overleftarrow{ac})_{min}:0$', 'color': 'green', 'style': ':'},
                          {'value': 3.0, 'label': r'$l(\overleftarrow{ac})_{max}:3\sum{T}$', 'color': 'green', 'style': '-.'}
                      ],
                      x_tick_labels=[(0.0, r'$0$'), (1.0, r'$\sum{T}$'), (2.0, r'$2\sum{T}$'), (3.0, r'$3\sum{T}$')],
                      x_axis_range=(-0.1, 3.1), mean_postfix=r'$\sum{T}$',
                      show_title=False, fontsize=graph_font_size,
                      legend_loc='upper left', legend_bbox_to_anchor=(0.05, 0.97))
    print()




    print("=" * 25, 'RQ1 figures', "=" * 25)
    print("Draw scatter plot (x: oracle_chain_length, y: estimated_chain_legnth)")
    draw_scatter_plot(total_df, 'oracle_chain_length', 'estimated_chain_length', 
                      'Oracle Chain Length vs Estimated Chain Length', r'Oracle chain length $l(\overleftarrow{ac})$ (ms)', r'Estimated chain length $l(\widehat{\overleftarrow{ac}})$ (ms)',
                      show_plot=True, save_plot=True, output_dir=total_analysis_dir, add_y_equals_x=True, 
                      x_axis_range=(0, 1000), y_axis_range=(0, 1500),
                      x_tick_labels=[(0, '0'), (300, '300'), (600, '600'), (900, '900')],
                      y_tick_labels=[(0, '0'), (300, '300'), (600, '600'), (900, '900'), (1200, '1200'), (1500, '1500')],
                      x_tick_postfix='ms', y_tick_postfix='ms', show_title=False, fontsize=graph_font_size,
                      show_stats=False)
    print()

    print("Draw scatter plot (x: normalized_oracle_chain_length, y: normalized_estimated_chain_length)")
    draw_scatter_plot(total_df, 'normalized_oracle_chain_length', 'normalized_estimated_chain_length', 
                                             'Normalized Oracle Chain Length vs Normalized Estimated Chain Length', r'Oracle chain length $l(\overleftarrow{ac})$ normalized by $\sum{T}$', r'Estimated chain length $l(\widehat{\overleftarrow{ac}})$ normalized by $\sum{T}$',
                      show_plot=True, save_plot=True, output_dir=total_analysis_dir, 
                      add_y_equals_x=True, x_axis_range=(-0.1, 2.1), y_axis_range=(-0.1, 3.1),
                      x_tick_labels=[(0, r'$0$'), (1, r'$\sum{T}$'), (2, r'$2\sum{T}$')],
                      y_tick_labels=[(0, r'$0$'), (1, r'$\sum{T}$'), (2, r'$2\sum{T}$'), (3, r'$3\sum{T}$')],
                      reference_lines=[
                          {'type': 'vertical', 'value': 0, 'label': r'$l(\overleftarrow{ac})_{min}:0$', 'color': 'blue', 'style': ':'},
                          {'type': 'vertical', 'value': 2, 'label': r'$l(\overleftarrow{ac})_{max}:2\sum{T}$', 'color': 'blue', 'style': '-.'},
                          {'type': 'horizontal', 'value': 0, 'label': r'$l(\widehat{\overleftarrow{ac}})_{min}:0$', 'color': 'green', 'style': ':'},
                          {'type': 'horizontal', 'value': 3, 'label': r'$l(\widehat{\overleftarrow{ac}})_{max}:3\sum{T}$', 'color': 'green', 'style': '-.'}
                      ], show_title=False, fontsize=graph_font_size, show_stats=False,
                      legend_loc='lower right', legend_bbox_to_anchor=(0.95, 0.04))
    print()

    # print("Draw histogram plot (x: normalized_oracle_chain_length)")
    # draw_histogram_plot(total_df, 'normalized_oracle_chain_length', 
    #                     'Normalized Oracle Chain Length Distribution', 'Normalized Oracle Chain Length', 'Number of samples',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir)
    # print()

    # print("Draw histogram plot (x: normalized_estimated_chain_length)")
    # draw_histogram_plot(total_df, 'normalized_estimated_chain_length', 
    #                     'Normalized Estimated Chain Length Distribution', 'Normalized Estimated Chain Length', 'Number of samples',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir)
    # print()
    
    print("Draw histogram plot (x: error percentage)")
    
    # Calculate 90th percentile for error percentage
    error_data = total_df['error_percentage'].dropna()
    if not error_data.empty:
        percentile_90 = np.percentile(error_data, 90)
        print(f"90th Percentile: {percentile_90:.2f}%")
        
        # Add 90th percentile marker
        percentile_markers = [
            {'value': percentile_90, 'label': f'90th Percentile: {percentile_90:.1f}%', 'color': 'orange', 'style': '--'}
        ]
    else:
        percentile_markers = None
    
    draw_histogram_plot(total_df, 'error_percentage', 
                        'Error Percentage Distribution', r'Error percentage $\Delta/l(\overleftarrow{ac})$ (%)', 'Number of samples',
                      show_plot=True, save_plot=True, output_dir=total_analysis_dir,
                      x_axis_range=(0, 300), mean_postfix='%',
                      x_tick_labels=[(0, '0'), (100, '100'), (200, '200'), (300, '300')],
                      x_tick_postfix='%', show_title=False, fontsize=graph_font_size,
                      vertical_line_styles=percentile_markers)
    print()

    # table format analysis result of impact on error percentage from task_count, mean_period, mean_wcet_period.
    # Do multiple linear regression.
    
    
    print("=" * 25, 'RQ2 figures', "=" * 25)
    print("=" * 25, 'SPEARMAN CORRELATION ANALYSIS', "=" * 25)
    correlation_results = perform_spearman_correlation_analysis(total_df)
    print()
    print("=" * 25, 'MULTIPLE LINEAR REGRESSION ANALYSIS', "=" * 25)
    regression_results = perform_multiple_regression_table(total_df)
    print()
    # print("Draw scatter plot (x: task_count, y: error_percentage)")
    # draw_scatter_plot(total_df, 'task_count', 'error_percentage', 
    #                   'Task Count vs Error Percentage', 'Task Count', 'Error Percentage',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir)
    # print()
    
    # print("Draw scatter plot (x: min_period, y: error_percentage)")
    # draw_scatter_plot(total_df, 'min_period', 'error_percentage', 
    #                   'Min Period vs Error Percentage', 'Min Period', 'Error Percentage',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir)
    # print()
    
    # print("Draw scatter plot (x: mean_period, y: error_percentage)")
    # draw_scatter_plot(total_df, 'mean_period', 'error_percentage', 
    #                   'Mean Period vs Error Percentage', 'Mean Period', 'Error Percentage',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir)
    # print()
    
    # print("Draw scatter plot (x: max_period, y: error_percentage)")
    # draw_scatter_plot(total_df, 'max_period', 'error_percentage', 
    #                   'Max Period vs Error Percentage', 'Max Period', 'Error Percentage',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir)
    # print()
    
    # print("Draw scatter plot (x: min_wcet_period, y: error_percentage)")
    # draw_scatter_plot(total_df, 'min_wcet_period', 'error_percentage', 
    #                   'Min WCET/Period vs Error Percentage', 'Min WCET/Period', 'Error Percentage',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir)
    # print()
    
    # print("Draw scatter plot (x: mean_wcet_period, y: error_percentage)")
    # draw_scatter_plot(total_df, 'mean_wcet_period', 'error_percentage', 
    #                   'Mean WCET/Period vs Error Percentage', 'Mean WCET/Period', 'Error Percentage',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir)
    # print()
        
    # print("Draw scatter plot (x: max_wcet_period, y: error_percentage)")
    # draw_scatter_plot(total_df, 'max_wcet_period', 'error_percentage', 
    #                   'Max WCET/Period vs Error Percentage', 'Max WCET/Period', 'Error Percentage',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir)
    # print()

    
    


if __name__ == "__main__":
    main()