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


def draw_scatter_plot(total_df: pd.DataFrame, x_col: str, y_col: str, title: str, x_label: str, y_label: str, show_plot: bool = True, save_plot: bool = True, output_dir: Path = None, add_y_equals_x: bool = False, x_axis_range: tuple = None, y_axis_range: tuple = None, x_tick_labels: List[tuple] = None, y_tick_labels: List[tuple] = None, reference_lines: List[dict] = None, show_stats: bool = True, x_tick_postfix: str = None, y_tick_postfix: str = None, show_title: bool = True, fontsize: int = 14, legend_loc: str = None, legend_bbox_to_anchor: tuple = None, category_cols: List[str] = None, category_colors: List[str] = None, category_markers: List[str] = None, category_alpha: float = 0.6, category_size: int = 15, show_category_trends: bool = False, trend_line_style: str = '--', trend_line_alpha: float = 0.8, category_labels: List[str] = None):
    """
    Draw a scatter plot for the given columns with optional multi-category support
    
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
        category_cols: List of column names to use for categorization (e.g., ['max_period', 'max_wcet_period'])
        category_colors: List of colors for different category combinations (optional, auto-generated if None)
        category_markers: List of markers for different category combinations (optional, auto-generated if None)
        category_alpha: Alpha transparency for scatter points (default 0.6)
        category_size: Size of scatter points (default 15)
        show_category_trends: Whether to show trend lines for each category combination (default False)
        trend_line_style: Style for trend lines ('-', '--', '-.', ':', etc.) (default '--')
        trend_line_alpha: Alpha transparency for trend lines (default 0.8)
        category_labels: List of custom labels for category columns (e.g., [r'Period $T$ (ms)', r'Utility $C/T$'])
    """
    # Filter out NaN values
    valid_data = total_df.dropna(subset=[x_col, y_col])
    
    if valid_data.empty:
        print(f"No valid data for scatter plot: {x_col} vs {y_col}")
        return None
    
    # Create the plot
    plt.figure(figsize=(8, 8))
    
    # Handle multi-category plotting
    if category_cols is not None and len(category_cols) > 0:
        # Filter out NaN values for category columns as well
        category_data = valid_data.dropna(subset=category_cols)
        
        if category_data.empty:
            print(f"No valid data for scatter plot with categories: {x_col} vs {y_col}")
            return None
        
        # Get unique combinations of category values
        category_combinations = category_data[category_cols].drop_duplicates().sort_values(category_cols)
        num_combinations = len(category_combinations)
        
        # Generate colors and markers if not provided
        if category_colors is None:
            # Use a color palette that works well for multiple categories
            colors = plt.cm.Set3(np.linspace(0, 1, num_combinations))
            category_colors = [colors[i] for i in range(num_combinations)]
        else:
            # Ensure we have enough colors by cycling through the provided ones
            category_colors = [category_colors[i % len(category_colors)] for i in range(num_combinations)]
        
        if category_markers is None:
            # Use different markers for different combinations
            markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
            category_markers = [markers[i % len(markers)] for i in range(num_combinations)]
        else:
            # Ensure we have enough markers by cycling through the provided ones
            category_markers = [category_markers[i % len(category_markers)] for i in range(num_combinations)]
        
        # Plot each category combination
        legend_handles = []
        for combo_idx, (_, row) in enumerate(category_combinations.iterrows()):
            # Create mask for this combination
            mask = True
            for col in category_cols:
                mask = mask & (category_data[col] == row[col])
            
            subset_data = category_data[mask]
            
            if len(subset_data) > 0:
                # Create label for this combination
                label_parts = []
                for i, col in enumerate(category_cols):
                    if category_labels is not None and i < len(category_labels):
                        # Use custom label if provided
                        if row[col]>1:
                            # if the value is greater than 1, only show integer part
                            label_parts.append(f"{category_labels[i]}={int(row[col])}")
                        else:
                            label_parts.append(f"{category_labels[i]}={row[col]}")
                    else:
                        # Use column name as fallback
                        label_parts.append(f"{col}={row[col]}")
                label = ", ".join(label_parts)
                
                # Calculate correlation for this category
                if len(subset_data) > 1:
                    try:
                        correlation, p_value = stats.pearsonr(subset_data[x_col], subset_data[y_col])
                        # Add correlation info to the label
                        enhanced_label = f"{label}\n(r={correlation:.2f}, p={p_value:.2e})"
                    except (ValueError, np.linalg.LinAlgError):
                        # Handle cases where correlation cannot be calculated
                        enhanced_label = f"{label}\n(r=N/A, p=N/A)"
                else:
                    enhanced_label = label
                
                # Plot this category
                scatter = plt.scatter(subset_data[x_col], subset_data[y_col], 
                                    alpha=category_alpha, s=category_size, 
                                    facecolors='none', marker=category_markers[combo_idx], 
                                    edgecolors=category_colors[combo_idx], linewidth=2.5,
                                    label=enhanced_label)
                legend_handles.append(scatter)
                
                # Add trend line for this category if requested (without legend)
                if show_category_trends and len(subset_data) > 1:
                    try:
                        # Calculate linear regression for this category
                        slope, intercept, r_value, p_value, std_err = stats.linregress(subset_data[x_col], subset_data[y_col])
                        
                        # Generate x values for trend line
                        x_trend = np.linspace(subset_data[x_col].min(), subset_data[x_col].max(), 100)
                        y_trend = slope * x_trend + intercept
                        
                        # Plot trend line with same color as the category (no label)
                        plt.plot(x_trend, y_trend, color=category_colors[combo_idx], 
                                linestyle=trend_line_style, alpha=trend_line_alpha, 
                                linewidth=2)
                    except (ValueError, np.linalg.LinAlgError):
                        # Skip trend line if linear regression cannot be calculated
                        pass
    else:
        # Single category plotting (original behavior)
        plt.scatter(valid_data[x_col], valid_data[y_col], alpha=category_alpha, s=category_size, 
                   facecolors='none', edgecolors='blue', linewidth=1)
    
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
    # Use category_data if available, otherwise use valid_data
    if category_cols is not None and len(category_cols) > 0 and 'category_data' in locals():
        data_for_correlation = category_data
    else:
        data_for_correlation = valid_data
    correlation, p_value = stats.pearsonr(data_for_correlation[x_col], data_for_correlation[y_col])
    
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
    
    # Add legend if we have any lines or categories (trend lines don't get legend entries)
    has_legend_items = (reference_lines is not None or add_y_equals_x or 
                       (category_cols is not None and len(category_cols) > 0))
    
    if has_legend_items:
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


def draw_multi_category_scatter_plot(total_df: pd.DataFrame, x_col: str, y_col: str, category_cols: List[str], 
                                   title: str, x_label: str, y_label: str, show_plot: bool = True, 
                                   save_plot: bool = True, output_dir: Path = None, show_category_trends: bool = False,
                                   trend_line_style: str = '--', trend_line_alpha: float = 0.8, category_labels: List[str] = None, **kwargs):
    """
    Convenience function to draw a scatter plot with multiple categories
    
    This function is a wrapper around draw_scatter_plot that automatically handles
    the multi-category visualization with different colors and shapes for each
    unique combination of the specified category columns.
    
    Args:
        total_df: DataFrame containing the data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        category_cols: List of column names to use for categorization (e.g., ['max_period', 'max_wcet_period'])
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        show_plot: Whether to show the plot as popup
        save_plot: Whether to save the plot automatically
        output_dir: Directory to save the plot
        show_category_trends: Whether to show trend lines for each category combination
        trend_line_style: Style for trend lines ('-', '--', '-.', ':', etc.)
        trend_line_alpha: Alpha transparency for trend lines
        category_labels: List of custom labels for category columns (e.g., [r'Period $T$ (ms)', r'Utility $C/T$'])
        **kwargs: Additional arguments passed to draw_scatter_plot
    
    Example:
        draw_multi_category_scatter_plot(
            total_df, 'task_count', 'error_percentage', 
            ['max_period', 'max_wcet_period'],
            'Task Count vs Error Percentage by Period and Utilization',
            'Task Count', 'Error Percentage (%)'
        )
    """
    return draw_scatter_plot(
        total_df=total_df,
        x_col=x_col,
        y_col=y_col,
        title=title,
        x_label=x_label,
        y_label=y_label,
        show_plot=show_plot,
        save_plot=save_plot,
        output_dir=output_dir,
        category_cols=category_cols,
        show_category_trends=show_category_trends,
        trend_line_style=trend_line_style,
        trend_line_alpha=trend_line_alpha,
        category_labels=category_labels,
        **kwargs
    )


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

    # print("Draw histogram plot (x: normalized_oracle_chain_length)")
    # draw_histogram_plot(total_df, 'normalized_oracle_chain_length', 
    #                     'Normalized Oracle Chain Length Distribution', r'Oracle Chain Length $l(\overleftarrow{ac})$ normalized by $\sum{T}$', 'Number of samples',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir, 
    #                   vertical_line_styles=[
    #                       {'value': 0.0, 'label': r'$l(\overleftarrow{ac})_{min}:0$', 'color': 'blue', 'style': ':'},
    #                       {'value': 2.0, 'label': r'$l(\overleftarrow{ac})_{max}:2\sum{T}$', 'color': 'blue', 'style': '-.'}
    #                   ],
    #                   x_tick_labels=[(0.0, r'$0$'), (1.0, r'$\sum{T}$'), (2.0, r'$2\sum{T}$'), (3.0, r'$3\sum{T}$')],
    #                   x_axis_range=(-0.1, 3.1), mean_postfix=r'$\sum{T}$',
    #                   show_title=False, fontsize=graph_font_size)
    # print()
    
    
    # print("Draw histogram plot (x: normalized_estimated_chain_length)")
    # draw_histogram_plot(total_df, 'normalized_estimated_chain_length', 
    #                     'Normalized Estimated Chain Length Distribution', r'Estimated Chain Length $l(\widehat{\overleftarrow{ac}})$ normalized by $\sum{T}$', 'Number of samples',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir, 
    #                   vertical_line_styles=[
    #                       {'value': 0.0, 'label': r'$l(\overleftarrow{ac})_{min}:0$', 'color': 'green', 'style': ':'},
    #                       {'value': 3.0, 'label': r'$l(\overleftarrow{ac})_{max}:3\sum{T}$', 'color': 'green', 'style': '-.'}
    #                   ],
    #                   x_tick_labels=[(0.0, r'$0$'), (1.0, r'$\sum{T}$'), (2.0, r'$2\sum{T}$'), (3.0, r'$3\sum{T}$')],
    #                   x_axis_range=(-0.1, 3.1), mean_postfix=r'$\sum{T}$',
    #                   show_title=False, fontsize=graph_font_size,
    #                   legend_loc='upper left', legend_bbox_to_anchor=(0.05, 0.97))
    # print()




    print("=" * 25, 'RQ1 figures', "=" * 25)
    # print("Draw scatter plot (x: oracle_chain_length, y: estimated_chain_legnth)")
    # draw_scatter_plot(total_df, 'oracle_chain_length', 'estimated_chain_length', 
    #                   'Oracle Chain Length vs Estimated Chain Length', r'Oracle Chain Length $l(\overleftarrow{ac})$', r'Estimated Chain Length $l(\widehat{\overleftarrow{ac}})$',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir, add_y_equals_x=True, 
    #                   x_axis_range=(0, 1500), y_axis_range=(0, 2500),
    #                   x_tick_labels=[(0, '0'), (500, '500'), (1000, '1000'), (1500, '1500')],
    #                   y_tick_labels=[(0, '0'), (500, '500'), (1000, '1000'), (1500, '1500'), (2000, '2000'), (2500, '2500')],
    #                   x_tick_postfix='ms', y_tick_postfix='ms', show_title=False, fontsize=graph_font_size,
    #                   show_stats=False)
    # print()

    # print("Draw scatter plot (x: normalized_oracle_chain_length, y: normalized_estimated_chain_length)")
    # draw_scatter_plot(total_df, 'normalized_oracle_chain_length', 'normalized_estimated_chain_length', 
    #                                          'Normalized Oracle Chain Length vs Normalized Estimated Chain Length', r'Oracle Chain Length $l(\overleftarrow{ac})$ normalized by $\sum{T}$', r'Estimated Chain Length $l(\widehat{\overleftarrow{ac}})$ normalized by $\sum{T}$',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir, 
    #                   add_y_equals_x=True, x_axis_range=(-0.1, 2.1), y_axis_range=(-0.1, 3.1),
    #                   x_tick_labels=[(0, r'$0$'), (1, r'$\sum{T}$'), (2, r'$2\sum{T}$')],
    #                   y_tick_labels=[(0, r'$0$'), (1, r'$\sum{T}$'), (2, r'$2\sum{T}$'), (3, r'$3\sum{T}$')],
    #                   reference_lines=[
    #                       {'type': 'vertical', 'value': 0, 'label': r'$l(\overleftarrow{ac})_{min}:0$', 'color': 'blue', 'style': ':'},
    #                       {'type': 'vertical', 'value': 2, 'label': r'$l(\overleftarrow{ac})_{max}:2\sum{T}$', 'color': 'blue', 'style': '-.'},
    #                       {'type': 'horizontal', 'value': 0, 'label': r'$l(\widehat{\overleftarrow{ac}})_{min}:0$', 'color': 'green', 'style': ':'},
    #                       {'type': 'horizontal', 'value': 3, 'label': r'$l(\widehat{\overleftarrow{ac}})_{max}:3\sum{T}$', 'color': 'green', 'style': '-.'}
    #                   ], show_title=False, fontsize=graph_font_size, show_stats=False,
    #                   legend_loc='lower right', legend_bbox_to_anchor=(0.95, 0.04))
    # print()

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
    
    # print("Draw histogram plot (x: error percentage)")
    # draw_histogram_plot(total_df, 'error_percentage', 
    #                     'Error Percentage Distribution', r'Error Percentage $\Delta/l(\overleftarrow{ac})$', 'Number of samples',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir,
    #                   x_axis_range=(0, 400), mean_postfix='%',
    #                   x_tick_labels=[(0, '0'), (100, '100'), (200, '200'), (300, '300'), (400, '400')],
    #                   x_tick_postfix='%', show_title=False, fontsize=graph_font_size)
    # print()
    
    print("=" * 25, 'RQ2 figures', "=" * 25)
    # print("Draw scatter plot (x: task_count, y: error_percentage)")
    # draw_scatter_plot(total_df, 'task_count', 'error_percentage', 
    #                   'Task Count vs Error Percentage', 'Task Count', 'Error Percentage',
    #                   show_plot=True, save_plot=True, output_dir=total_analysis_dir)
    # print()
    
    print("Draw multi-category scatter plot (x: task_count, y: error_percentage, categories: max_period, max_wcet_period)")
    draw_multi_category_scatter_plot(
        total_df, 'task_count', 'error_percentage', 
        ['max_period', 'max_wcet_period'],
        'Task Count vs Error Percentage by Max Period and Max Utilization',
        r'Number of tasks in the chain $|E|$', r'Error percentage $\Delta/l(\overleftarrow{ac})$ (%)',
        show_plot=True, save_plot=True, output_dir=total_analysis_dir,
        legend_loc='upper right', fontsize=18,
        category_colors=['red', 'blue', 'green', 'orange'],
        category_markers=['o', 's', '^', 'v'],
        category_size=40,
        show_category_trends=True,
        trend_line_style='--',
        trend_line_alpha=0.7,
        show_stats=False,
        category_labels=[r'$T$ (ms)', r'$C/T$'],
        show_title=False
    )
    print()
    
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