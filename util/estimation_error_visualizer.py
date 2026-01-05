#!/usr/bin/env python3
"""
Estimation Error Visualizer

This module provides visualization capabilities for cause-effect chain estimation results
using scatter plots to show the relationship between oracle and estimated chain lengths.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import statistics


@dataclass
class ScatterPlotConfig:
    """Configuration for scatter plot visualization"""
    figure_width: int = 10
    figure_height: int = 8
    point_size: int = 50
    alpha: float = 0.7
    title: str = "Cause-Effect Chain Estimation Error Analysis"
    xlabel: str = "Oracle Chain Length (ms)"
    ylabel: str = "Estimated Chain Length (ms)"
    grid_alpha: float = 0.3
    reference_line_color: str = 'red'
    reference_line_style: str = '--'
    reference_line_alpha: float = 0.8


class EstimationErrorVisualizer:
    """Visualizer class for estimation error analysis"""
    
    def __init__(self, config: Optional[ScatterPlotConfig] = None):
        """
        Initialize the visualizer
        
        Args:
            config: Configuration for scatter plot visualization
        """
        self.config = config or ScatterPlotConfig()
        self._last_figure: Optional[plt.Figure] = None
    
    def create_scatter_plot(self, evaluation_results: List[Dict[str, Any]], 
                          profile_name: str = "unknown") -> plt.Figure:
        """
        Create a scatter plot from evaluation results
        
        Args:
            evaluation_results: List of dictionaries containing evaluation results
            profile_name: Name of the profile for the plot title
            
        Returns:
            matplotlib Figure object
        """
        if not evaluation_results:
            raise ValueError("No evaluation results provided")
        
        # Filter out None values
        valid_results = [
            result for result in evaluation_results 
            if result['oracle_chain_length'] is not None and result['estimated_chain_length'] is not None
        ]
        
        if not valid_results:
            raise ValueError("No valid data points found in evaluation results")
        
        # Extract data
        oracle_lengths = [result['oracle_chain_length'] for result in valid_results]
        estimated_lengths = [result['estimated_chain_length'] for result in valid_results]
        errors = [result['error'] for result in valid_results]
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(self.config.figure_width, self.config.figure_height))
        
        # Create scatter plot
        scatter = ax.scatter(oracle_lengths, estimated_lengths, 
                           s=self.config.point_size, alpha=self.config.alpha,
                           c=errors, cmap='viridis', edgecolors='black', linewidth=0.5)
        
        # Add colorbar for error values
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Error (Estimated - Oracle) (ms)', rotation=270, labelpad=15)
        
        # Calculate plot limits
        min_val = min(min(oracle_lengths), min(estimated_lengths))
        max_val = max(max(oracle_lengths), max(estimated_lengths))
        margin = (max_val - min_val) * 0.1  # 10% margin
        
        # Set axis limits
        ax.set_xlim(min_val - margin, max_val + margin)
        ax.set_ylim(min_val - margin, max_val + margin)
        
        # Add y=x reference line (perfect estimation)
        ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 
               color=self.config.reference_line_color, linestyle=self.config.reference_line_style, 
               alpha=self.config.reference_line_alpha, linewidth=2)
        
        # Add grid
        ax.grid(True, alpha=self.config.grid_alpha)
        
        # Set labels and title
        ax.set_xlabel(self.config.xlabel, fontsize=12)
        ax.set_ylabel(self.config.ylabel, fontsize=12)
        ax.set_title(f"{self.config.title} - {profile_name}", fontsize=14, weight='bold', pad=20)
        
        # Add legend
        # ax.legend(loc='upper left')
        
        # Add statistics annotations
        self._add_statistics_annotations(ax, oracle_lengths, estimated_lengths, errors)
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
    
    def _add_statistics_annotations(self, ax: plt.Axes, oracle_lengths: List[float], 
                                  estimated_lengths: List[float], errors: List[float]) -> None:
        """
        Add statistical annotations to the plot
        
        Args:
            ax: Matplotlib axes to plot on
            oracle_lengths: List of oracle chain lengths
            estimated_lengths: List of estimated chain lengths
            errors: List of errors
        """
        # Calculate statistics
        oracle_mean = statistics.mean(oracle_lengths)
        oracle_std = statistics.stdev(oracle_lengths)
        estimated_mean = statistics.mean(estimated_lengths)
        estimated_std = statistics.stdev(estimated_lengths)
        error_mean = statistics.mean(errors)
        error_std = statistics.stdev(errors)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(oracle_lengths, estimated_lengths)[0, 1]
        
        # Create annotation text
        stats_text = f"""Statistics:
Oracle: {oracle_mean:.1f} ± {oracle_std:.1f} ms
Estimated: {estimated_mean:.1f} ± {estimated_std:.1f} ms
Error: {error_mean:.1f} ± {error_std:.1f} ms
Correlation: {correlation:.3f}
Data points: {len(oracle_lengths)}"""
        
        
        ax.text(0.7, 0.17, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    
    def create_error_histogram(self, evaluation_results: List[Dict[str, Any]], 
                             profile_name: str = "unknown", 
                             error_type: str = "absolute") -> plt.Figure:
        """
        Create a histogram of estimation errors
        
        Args:
            evaluation_results: List of dictionaries containing evaluation results
            profile_name: Name of the profile for the plot title
            error_type: Type of error to plot ("absolute" or "percentage")
            
        Returns:
            matplotlib Figure object
        """
        if not evaluation_results:
            raise ValueError("No evaluation results provided")
        
        # Determine which error field to use
        if error_type == "percentage":
            error_field = 'error_percentage'
            xlabel = 'Estimation Error Percentage (%)'
            title_suffix = 'Percentage'
        else:  # absolute
            error_field = 'error'
            xlabel = 'Estimation Error (Estimated - Oracle) (ms)'
            title_suffix = 'Distribution'
        
        # Filter out None values
        valid_results = [
            result for result in evaluation_results 
            if result[error_field] is not None
        ]
        
        if not valid_results:
            raise ValueError(f"No valid {error_type} error data found in evaluation results")
        
        # Extract errors
        errors = [result[error_field] for result in valid_results]
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(self.config.figure_width, self.config.figure_height))
        
        # Create histogram
        n, bins, patches = ax.hist(errors, bins=20, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add vertical line at error = 0
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        # Add mean line
        error_mean = statistics.mean(errors)
        ax.axvline(x=error_mean, color='green', linestyle='-', alpha=0.8, linewidth=2, 
                  label=f'Mean Error: {error_mean:.1f}{" %" if error_type == "percentage" else " ms"}')
        
        # Set labels and title
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Estimation Error {title_suffix} - {profile_name}', fontsize=14, weight='bold', pad=20)
        
        # Add grid
        ax.grid(True, alpha=self.config.grid_alpha)
        
        # Add legend (only for mean line)
        ax.legend()
        
        # Add statistics text
        error_std = statistics.stdev(errors)
        if error_type == "percentage":
            stats_text = f"""Error Statistics:
Mean: {error_mean:.1f} %
Std Dev: {error_std:.1f} %
Min: {min(errors):.1f} %
Max: {max(errors):.1f} %
Data points: {len(errors)}"""
        else:
            stats_text = f"""Error Statistics:
Mean: {error_mean:.1f} ms
Std Dev: {error_std:.1f} ms
Min: {min(errors):.1f} ms
Max: {max(errors):.1f} ms
Data points: {len(errors)}"""
        
        ax.text(0.07, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
    
    def save_plot(self, fig: plt.Figure, output_file: str) -> None:
        """
        Save the plot to a file
        
        Args:
            fig: Matplotlib Figure object
            output_file: File path to save the plot
        """
        try:
            # Ensure output directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with high DPI for quality
            fig.savefig(output_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            # print(f"Plot saved successfully to: {output_file}")
            
        except Exception as e:
            print(f"Error saving plot: {e}")
            raise
    
    def display_plot(self, fig: plt.Figure) -> None:
        """
        Display the plot interactively
        
        Args:
            fig: Matplotlib Figure object
        """
        try:
            plt.figure(fig.number)
            plt.show()
        except Exception as e:
            print(f"Error displaying plot: {e}")
            raise
    
    def generate_and_save_plots(self, evaluation_results: List[Dict[str, Any]], 
                              profile_name: str, output_dir: str = "evaluation_results_plots") -> None:
        """
        Generate and save both scatter plot and error histogram
        
        Args:
            evaluation_results: List of dictionaries containing evaluation results
            profile_name: Name of the profile for the plot titles
            output_dir: Directory to save the plots
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate scatter plot
        # print(f"Generating scatter plot for {profile_name}...")
        scatter_fig = self.create_scatter_plot(evaluation_results, profile_name)
        scatter_file = output_path / f"scatter_plot_{profile_name}.png"
        self.save_plot(scatter_fig, str(scatter_file))
        
        # # Generate error histogram
        # # print(f"Generating error histogram for {profile_name}...")
        # histogram_fig = self.create_error_histogram(evaluation_results, profile_name, "absolute")
        # histogram_file = output_path / f"error_histogram_{profile_name}.png"
        # self.save_plot(histogram_fig, str(histogram_file))

        # Generate error percentage histogram
        # print(f"Generating error percentage histogram for {profile_name}...")
        error_percentage_fig = self.create_error_histogram(evaluation_results, profile_name, "percentage")
        error_percentage_file = output_path / f"error_percentage_histogram_{profile_name}.png"
        self.save_plot(error_percentage_fig, str(error_percentage_file))
        
        # Store the last figure for potential display
        self._last_figure = scatter_fig
        
        # print(f"Plots saved to {output_dir}/")
    
    def display_last_plot(self) -> None:
        """
        Display the last generated plot interactively
        
        Raises:
            RuntimeError: If no plot has been generated yet
        """
        if not hasattr(self, '_last_figure') or self._last_figure is None:
            raise RuntimeError("No plot has been generated yet. Call generate_and_save_plots first.")
        
        self.display_plot(self._last_figure)
    
    def clear_last_figure(self) -> None:
        """
        Clear the reference to the last figure to free memory
        """
        if hasattr(self, '_last_figure') and self._last_figure is not None:
            self._last_figure = None


def main():
    """Test function for the estimation error visualizer"""
    print("Estimation Error Visualizer")
    print("=" * 40)
    
    # Create visualizer instance
    visualizer = EstimationErrorVisualizer()
    
    # Test configuration
    print(f"Default figure size: {visualizer.config.figure_width}x{visualizer.config.figure_height}")
    print(f"Point size: {visualizer.config.point_size}")
    print(f"Alpha: {visualizer.config.alpha}")
    
    print("\nVisualizer initialized successfully!")
    print("Ready to create scatter plots and histograms from evaluation results.")


if __name__ == "__main__":
    main()
