#!/usr/bin/env python3
"""
Cause-Effect Chain Visualizer

This module provides visualization capabilities for cause-effect chain simulation results
using Gantt charts to show task execution timelines.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
from pathlib import Path


@dataclass
class GanttChartConfig:
    """Configuration for Gantt chart visualization"""
    figure_width: int = 12
    figure_height: int = 8
    task_height: float = 0.6
    task_spacing: float = 0.2
    colors: Dict[str, str] = None
    title: str = "Cause-Effect Chain Simulation Gantt Chart"
    xlabel: str = "Time (ms)"
    ylabel: str = "Tasks"


class CauseEffectChainVisualizer:
    """Visualizer class for cause-effect chain simulation results"""
    
    def __init__(self, config: Optional[GanttChartConfig] = None):
        """
        Initialize the visualizer
        
        Args:
            config: Configuration for Gantt chart visualization
        """
        self.config = config or GanttChartConfig()
        self._setup_default_colors()
        self._last_figure: Optional[plt.Figure] = None
    
    def _setup_default_colors(self):
        """Setup default color scheme for different event types"""
        if self.config.colors is None:
            self.config.colors = {
                'release': '#FF6B6B',      # Red for release events
                'read-event': '#4ECDC4',   # Teal for read events
                'write-event': '#45B7D1',  # Blue for write events
                'execution': '#96CEB4'     # Green for execution periods
            }
    
    def create_gantt_chart(self, simulation_results: Dict[str, Any], 
                          estimation_result: Optional[Any] = None) -> plt.Figure:
        """
        Create a Gantt chart from simulation results
        
        Args:
            simulation_results: Results from the simulator
            estimation_result: Optional estimation results for cause-effect chains
            
        Returns:
            matplotlib Figure object
        """
        if not simulation_results:
            raise ValueError("No simulation results provided")
        
        # Get task names and determine chart dimensions
        task_names = list(simulation_results.keys())
        num_tasks = len(task_names)
        
        if num_tasks == 0:
            raise ValueError("No tasks found in simulation results")
        
        # Calculate simulation duration from all events
        max_time = 0
        for task_result in simulation_results.values():
            if hasattr(task_result, 'events') and task_result.events:
                max_time = max(max_time, max(event.timestamp for event in task_result.events))
        
        # Adjust figure size based on simulation duration and number of tasks
        duration_factor = max(1, max_time / 1000)  # Base on 1000ms
        task_factor = max(1, num_tasks / 5)        # Base on 5 tasks
        
        fig_width = self.config.figure_width * duration_factor
        fig_height = self.config.figure_height * task_factor
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Set up the chart
        self._setup_chart_axes(ax, task_names, max_time)
        
        # Plot each task's timeline
        for i, task_name in enumerate(task_names):
            task_result = simulation_results[task_name]
            y_position = len(task_names) - 1 - i  # Top to bottom
            
            # Plot execution blocks (read-event to write-event)
            self._plot_execution_blocks(task_result, ax, y_position, task_name, max_time)
            
            # Plot release time markers
            self._plot_release_markers(task_result, ax, y_position, task_name)
        
        # Add cause-effect chain information if available
        if estimation_result:
            self._add_cause_effect_annotations(ax, estimation_result, max_time, num_tasks)
        
        # Customize the chart
        self._customize_chart(fig, ax, max_time, num_tasks)
        
        return fig
    
    def _setup_chart_axes(self, ax: plt.Axes, task_names: List[str], max_time: int):
        """Setup the chart axes with proper labels and limits"""
        # Set y-axis (tasks)
        ax.set_yticks(range(len(task_names)))
        # Reverse labels so task_names[0] (task1) appears at top, task_names[-1] (taskN) at bottom
        ax.set_yticklabels(task_names[::-1])
        ax.set_ylim(-0.5, len(task_names) - 0.5)
        
        # Set x-axis (time)
        ax.set_xlim(0, max_time)
        ax.set_xlabel(self.config.xlabel)
        ax.set_ylabel(self.config.ylabel)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='x')
    
    def _plot_execution_blocks(self, task_result: Any, ax: plt.Axes, y_position: float, task_name: str, max_time: int):
        """Plot execution blocks from read-event to write-event"""
        if not hasattr(task_result, 'events') or not task_result.events:
            return
        
        # Group events by type for easier processing
        read_events = [e for e in task_result.events if e.event_type == 'read-event']
        write_events = [e for e in task_result.events if e.event_type == 'write-event']
        
        # Sort events by timestamp
        read_events.sort(key=lambda x: x.timestamp)
        write_events.sort(key=lambda x: x.timestamp)
        
        # Plot execution blocks
        for read_event, write_event in zip(read_events, write_events):
            start_time = read_event.timestamp
            end_time = write_event.timestamp
            duration = end_time - start_time
            
            # Create execution block rectangle
            rect = patches.Rectangle(
                (start_time, y_position - self.config.task_height/2),
                duration,
                self.config.task_height,
                facecolor=self.config.colors['execution'],
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # # Add duration label if block is wide enough
            # if duration > max_time * 0.05:  # Only label if block is >5% of total time
            #     ax.text(start_time + duration/2, y_position, 
            #            f'{duration}ms', ha='center', va='center', 
            #            fontsize=8, weight='bold')
    
    def _plot_release_markers(self, task_result: Any, ax: plt.Axes, y_position: float, task_name: str):
        """Plot release time markers"""
        if not hasattr(task_result, 'events') or not task_result.events:
            return
        
        # Get release events
        release_events = [e for e in task_result.events if e.event_type == 'release']
        
        for release_event in release_events:
            release_time = release_event.timestamp
            
            # Create a thin vertical line marker for release time
            y_min = y_position - self.config.task_height/2
            y_max = y_position + self.config.task_height/2
            ax.plot([release_time, release_time], [y_min, y_max],
                   color=self.config.colors['release'], linewidth=3, alpha=0.8)
    
    def _customize_chart(self, fig: plt.Figure, ax: plt.Axes, max_time: int, num_tasks: int):
        """Customize the chart appearance"""
        # Set title
        ax.set_title(self.config.title, fontsize=14, weight='bold', pad=20)
        
        # Adjust layout
        fig.tight_layout()
        
        # Add legend
        legend_elements = [
            patches.Patch(color=self.config.colors['execution'], label='Task Execution'),
            patches.Patch(color=self.config.colors['release'], label='Release Time'),
            patches.Patch(color='blue', label='Oracle Chain'),
            patches.Patch(color='red', label='Estimated Chain')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    def _add_cause_effect_annotations(self, ax: plt.Axes, estimation_result: Any, 
                                    max_time: int, num_tasks: int) -> None:
        """
        Add cause-effect chain annotations to the Gantt chart
        
        Args:
            ax: Matplotlib axes to plot on
            estimation_result: Estimation results containing cause-effect relationships
            max_time: Maximum simulation time
            num_tasks: Number of tasks
        """
        if not estimation_result or not hasattr(estimation_result, 'oracle_chains'):
            return
        
        # Get task names for y-position mapping
        # Gantt chart shows tasks from top (task1) to bottom (taskN)
        task_names = list(ax.get_yticklabels())
        # Since labels are reversed, the mapping should be: task_names[0] (taskN) at top (0), task_names[-1] (task1) at bottom (len-1)
        task_name_to_y = {task_names[i].get_text(): i for i in range(len(task_names))}
        
        # Process each oracle chain
        for chain in estimation_result.oracle_chains:
            if not chain.task_sequence or len(chain.task_sequence) < 2:
                continue
            
            # print(f"  Visualizing oracle chain: {chain.chain_id}")
            self._visualize_single_chain(ax, chain, task_name_to_y, 'oracle', estimation_result)
        
        # Process each estimated chain
        if hasattr(estimation_result, 'estimated_chains') and estimation_result.estimated_chains:
            for chain in estimation_result.estimated_chains:
                if not chain.task_sequence or len(chain.task_sequence) < 2:
                    continue
                
                # print(f"  Visualizing estimated chain: {chain.chain_id}")
                self._visualize_single_chain(ax, chain, task_name_to_y, 'estimated', estimation_result)
    
    def _visualize_single_chain(self, ax: plt.Axes, chain: Any, task_name_to_y: Dict[str, int], 
                               chain_type: str, estimation_result: Any) -> None:
        """
        Visualize a single cause-effect chain
        
        Args:
            ax: Matplotlib axes to plot on
            chain: The chain to visualize
            task_name_to_y: Mapping of task names to y positions
            chain_type: Type of chain ('oracle' or 'estimated')
            estimation_result: Estimation results containing z event
        """
        # Calculate vertical offset based on task height
        task_height = self.config.task_height
        offset = task_height / 6
        
        # Set colors and offsets based on chain type
        if chain_type == 'oracle':
            arrow_color = 'blue'      # For both intra-task and z arrows
            dependency_color = 'blue' # Same as arrow_color
            y_offset = offset        # Oracle chain arrows below task center
        else:  # estimated
            arrow_color = 'red'   # For both intra-task and z arrows
            dependency_color = 'red' # Same as arrow_color
            y_offset = -offset         # Estimated chain arrows above task center
        
        # First, visualize the z event connection if available
        if hasattr(chain, 'z_event') and chain.z_event is not None:
            z_event = chain.z_event
            first_task_name = chain.task_sequence[0].task_name
            
            if first_task_name in task_name_to_y:
                first_task_y = task_name_to_y[first_task_name] + y_offset
                z_time = z_event.timestamp
                
                # Get the first task's chain start read-event time
                first_task_read_time = chain.task_sequence[0].read_event.timestamp
                
                # Draw dependency arrow from z to first task's chain start
                ax.annotate('', 
                           xy=(first_task_read_time, first_task_y),  # Arrow head at chain start read-event
                           xytext=(z_time, first_task_y),  # Arrow tail at z event
                           arrowprops=dict(arrowstyle='->', 
                                         color=dependency_color, 
                                         linewidth=2, 
                                         alpha=0.8,
                                         linestyle=':'))  # Dotted line for z dependency
                
                # Add z event marker
                ax.plot(z_time, first_task_y, 'o', color=dependency_color, markersize=8, alpha=0.8)
                
                # print(f"    Z event: {z_time}ms → First task chain start: {first_task_read_time}ms")
        
        # Visualize the z' event connection if available
        if hasattr(chain, 'z_prime_event') and chain.z_prime_event is not None:
            z_prime_event = chain.z_prime_event
            last_task_name = chain.task_sequence[-1].task_name
            
            if last_task_name in task_name_to_y:
                last_task_y = task_name_to_y[last_task_name] + y_offset
                z_prime_time = z_prime_event.timestamp
                
                # Get the last task's chain write-event time
                last_task_write_time = chain.task_sequence[-1].write_event.timestamp
                
                # Draw dependency arrow from last task's write-event to z'
                ax.annotate('', 
                           xy=(z_prime_time, last_task_y),  # Arrow head at z' event
                           xytext=(last_task_write_time, last_task_y),  # Arrow tail at last task write-event
                           arrowprops=dict(arrowstyle='->', 
                                         color=dependency_color, 
                                         linewidth=2, 
                                         alpha=0.8,
                                         linestyle=':'))  # Dotted line for z' dependency
                
                # Add z' event marker
                ax.plot(z_prime_time, last_task_y, 's', color=dependency_color, markersize=8, alpha=0.8)
                
                # print(f"    Last task write: {last_task_write_time}ms → Z' event: {z_prime_time}ms")
        
        # Display chain length on the chart using pre-calculated value
        if hasattr(chain, 'chain_length') and chain.chain_length is not None:
            # print(f"    Total chain length: {chain.chain_length}ms (from z to z')")
            
            # Display chain length on the chart with chain type - right-aligned before legend
            ax.text(0.8, 0.98 - (0.05 if chain_type == 'estimated' else 0), 
                   f'{chain_type.title()} Chain Length: {chain.chain_length}ms', 
                   transform=ax.transAxes, fontsize=12, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Draw arrows for each task in the chain
        for i, task_pair in enumerate(chain.task_sequence):
            task_name = task_pair.task_name
            if task_name not in task_name_to_y:
                continue
            
            y_position = task_name_to_y[task_name] + y_offset
            read_time = task_pair.read_event.timestamp
            write_time = task_pair.write_event.timestamp
            
            # Draw arrow within the task (read-event to write-event)
            ax.annotate('', 
                       xy=(write_time, y_position),  # Arrow head at write-event
                       xytext=(read_time, y_position),  # Arrow tail at read-event
                       arrowprops=dict(arrowstyle='->', 
                                     color=arrow_color, 
                                     linewidth=2, 
                                     alpha=0.8))
            
            # Add small circle markers at read and write points
            ax.plot(read_time, y_position, 'ro', markersize=6, alpha=0.8)
            ax.plot(write_time, y_position, 'ro', markersize=6, alpha=0.8)
            
            # print(f"    Task {task_name}: {read_time}ms → {write_time}ms")
        
        # Draw arrows between adjacent tasks (write-event to read-event)
        for i in range(len(chain.task_sequence) - 1):
            current_task = chain.task_sequence[i]
            next_task = chain.task_sequence[i + 1]
            
            current_task_name = current_task.task_name
            next_task_name = next_task.task_name
            
            if current_task_name not in task_name_to_y or next_task_name not in task_name_to_y:
                continue
            
            current_y = task_name_to_y[current_task_name] + y_offset
            next_y = task_name_to_y[next_task_name] + y_offset
            
            # Start point: write-event of current task
            start_x = current_task.write_event.timestamp
            start_y = current_y
            
            # End point: read-event of next task
            end_x = next_task.read_event.timestamp
            end_y = next_y
            
            # Draw dependency arrow between tasks
            ax.annotate('', 
                       xy=(end_x, end_y),  # Arrow head at next task's read-event
                       xytext=(start_x, start_y),  # Arrow tail at current task's write-event
                       arrowprops=dict(arrowstyle='->', 
                                     color=dependency_color, 
                                     linewidth=2, 
                                     alpha=0.8,
                                     linestyle='--'))  # Dashed line for dependencies
            
            # print(f"    Dependency: {current_task_name} ({start_x}ms) → {next_task_name} ({end_x}ms)")
            

    
    def save_chart(self, fig: plt.Figure, output_file: str) -> None:
        """
        Save the chart to a file
        
        Args:
            fig: Matplotlib Figure object
            output_file: File path to save the chart
        """
        try:
            # Ensure output directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with high DPI for quality
            fig.savefig(output_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            # print(f"Chart saved successfully to: {output_file}")
            
        except Exception as e:
            print(f"Error saving chart: {e}")
            raise
    
    def display_chart(self, fig: plt.Figure) -> None:
        """
        Display the chart interactively
        
        Args:
            fig: Matplotlib Figure object
        """
        try:
            plt.figure(fig.number)
            plt.show()
        except Exception as e:
            print(f"Error displaying chart: {e}")
            raise
   
    
    def generate_and_save_gantt_chart(self, simulation_results: Dict[str, Any], 
                                    output_file: str, 
                                    estimation_result: Optional[Any] = None) -> None:
        """
        Generate and save a Gantt chart from simulation results
        
        Args:
            simulation_results: Results from the simulator
            output_file: File path to save the chart
            estimation_result: Optional estimation results for cause-effect chains
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        # Create the Gantt chart (now with estimation results if available)
        gantt_fig = self.create_gantt_chart(simulation_results, estimation_result)
        
        # Save the chart
        self.save_chart(gantt_fig, str(output_path))
        
        # Store the figure for potential display
        self._last_figure = gantt_fig
    
    def display_last_chart(self) -> None:
        """
        Display the last generated chart interactively
        
        Raises:
            RuntimeError: If no chart has been generated yet
        """
        if not hasattr(self, '_last_figure') or self._last_figure is None:
            raise RuntimeError("No chart has been generated yet. Call generate_and_save_gantt_chart first.")
        
        self.display_chart(self._last_figure)
    
    def clear_last_figure(self) -> None:
        """
        Clear the reference to the last figure to free memory
        """
        if hasattr(self, '_last_figure') and self._last_figure is not None:
            self._last_figure = None


def main():
    """Test function for the visualizer"""
    print("Cause-Effect Chain Visualizer")
    print("=" * 40)
    
    # Create visualizer instance
    visualizer = CauseEffectChainVisualizer()
    
    # Test configuration
    print(f"Default figure size: {visualizer.config.figure_width}x{visualizer.config.figure_height}")
    print(f"Task height: {visualizer.config.task_height}")
    print(f"Event colors: {list(visualizer.config.colors.keys())}")
    
    print("\nVisualizer initialized successfully!")
    print("Ready to create Gantt charts from simulation results.")


if __name__ == "__main__":
    main()
