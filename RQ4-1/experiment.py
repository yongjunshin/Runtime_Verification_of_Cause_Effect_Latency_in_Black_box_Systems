#!/usr/bin/env python3
"""
RQ4-1 Experiment: Estimation Time Analysis

This script measures the estimation time for cause-effect chain estimation
across different task counts (2-10) to analyze computational complexity.
"""

import sys
import time
import csv
import random
from pathlib import Path
from typing import Dict, List, Any
import statistics
import matplotlib.pyplot as plt
import numpy as np
import gc
import os

# Add parent directory to path to access util modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import the simulator, visualizer, and random profile generator
from util.cause_effect_chain_semantic_simulator import CauseEffectChainSimulator, TaskSimulationResult
from util.cause_effect_chain_estimator_new import CauseEffectChainEstimator
from util.random_profile_generator import RandomProfileGenerator


def export_estimation_time_results(results_data: List[Dict[str, Any]], output_file: str) -> None:
    """
    Export estimation time results to CSV file
    
    Args:
        results_data: List of dictionaries containing estimation time results
        output_file: Output CSV filename
    """
    # Create output directory if it doesn't exist
    output_dir = Path("estimation_time_results")
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / output_file
    
    # Define CSV headers
    headers = ['task_count', 'profile_id', 'simulation_repeat', 'estimation_time']
    
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            
            # Write header
            writer.writeheader()
            
            # Write data rows
            for result in results_data:
                writer.writerow(result)

    except Exception as e:
        print(f"Error exporting estimation time results: {e}")


def generate_whisker_plot(estimation_times_by_task_count: Dict[int, List[float]], output_file: str, 
                         show_title: bool = True, font_size: int = 12):
    """
    Generate a whisker plot showing estimation time distribution by task count
    
    Args:
        estimation_times_by_task_count: Dictionary mapping task count to list of estimation times
        output_file: Output filename for the plot
        show_title: Whether to display the title
        font_size: Global font size for all text elements
    """
    # Create output directory if it doesn't exist
    output_dir = Path("estimation_time_results")
    output_dir.mkdir(exist_ok=True)
    
    plot_path = output_dir / output_file
    
    # Prepare data for plotting
    task_counts = sorted(estimation_times_by_task_count.keys())
    estimation_times_data = [estimation_times_by_task_count[count] for count in task_counts]
    
    # Create the whisker plot
    plt.figure(figsize=(12, 8))
    
    # Create box plot
    box_plot = plt.boxplot(estimation_times_data, 
                          labels=task_counts,
                          patch_artist=True,
                          showfliers=True)
    
    # Color the boxes
    colors = plt.cm.viridis(np.linspace(0, 1, len(task_counts)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Get the primary axis before creating secondary axis
    ax1 = plt.gca()
    
    # Customize the primary axis
    ax1.set_xlabel('Number of tasks in the chain $|E|$', fontsize=font_size + 2)
    ax1.set_ylabel('Estimation Time (milliseconds)', fontsize=font_size + 2)
    if show_title:
        ax1.set_title('Estimation Time Distribution by Task Count', fontsize=font_size + 4, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Set tick label font sizes for primary axis
    ax1.tick_params(axis='x', labelsize=font_size)
    ax1.tick_params(axis='y', labelsize=font_size)
    
    # Ensure y1 axis doesn't go below 0 (estimation time cannot be negative)
    y1_min, y1_max = ax1.get_ylim()
    y1_min = max(0, y1_min)  # Don't go below 0
    
    # Round up the maximum estimation time to one decimal place
    y1_max_rounded = np.ceil(y1_max * 10) / 10  # Round up to 1 decimal place
    ax1.set_ylim(y1_min, y1_max_rounded)
    
    # Add secondary y-axis for estimations per second
    ax2 = ax1.twinx()
    
    # Make y2 identical to y1 - same limits, same ticks, but different labels
    ax2.set_ylim(y1_min, y1_max_rounded)
    ax2.set_yticks(ax1.get_yticks())
    
    # Create inverse tick labels for y2 (seconds/estimation_time_ms)
    y1_ticks = ax1.get_yticks()
    y2_tick_labels = []
    for tick in y1_ticks:
        if tick > 0:  # Avoid division by zero
            est_per_sec = 1000.0 / tick  # Convert ms to estimations per second
            y2_tick_labels.append(f'{est_per_sec:.0f}')
        else:
            y2_tick_labels.append('∞')
    
    ax2.set_yticklabels(y2_tick_labels)
    
    # Only difference: axis label
    ax2.set_ylabel('Estimations per Second', fontsize=font_size + 2)
    ax2.tick_params(axis='y', labelsize=font_size)
    
    # Ensure all tick labels have consistent font size
    for label in ax1.get_xticklabels():
        label.set_fontsize(font_size)
    for label in ax1.get_yticklabels():
        label.set_fontsize(font_size)
    for label in ax2.get_yticklabels():
        label.set_fontsize(font_size)
    
    
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Whisker plot saved to: {plot_path}")
    
    # Show the plot
    plt.show()


def main():
    """Main experiment controller for RQ4-1"""
    print("RQ4-1: Estimation Time Analysis")
    print("=" * 80)
    
    # ========================================
    # EXPERIMENT PARAMETER SETTINGS
    # ========================================
    
    # Task count range
    TASK_COUNTS = list(range(2, 21, 2))  # 2, 4, 6, 8, 10, ..., 20
    
    # Random profile generator parameters
    MIN_PERIOD = 20              # Minimum period in ms
    MAX_PERIOD = 100             # Maximum period in ms
    MIN_UTILIZATION = 0.1        # Minimum WCET/period ratio
    MAX_UTILIZATION = 0.9        # Maximum WCET/period ratio
    KEYWORD = None
    
    # Profile generation parameters
    NUM_PROFILES_PER_TASK_COUNT = 10    # Number of random profiles per task count
    SIMULATION_DURATION_MULTIPLIER = 3  # Simulation duration = MAX_PERIOD * TASK_COUNT * multiplier
    TIME_UNIT = "ms"             # Time unit
    
    # Simulation parameters
    SIMULATION_REPEATS = 10      # Number of simulation runs per profile
    BASE_SEED = 444              # Base random seed for reproducibility
    
    # ========================================
    # PRINT EXPERIMENT CONFIGURATION
    # ========================================
    
    print("=== Experiment Configuration ===")
    print(f"Task counts: {TASK_COUNTS}")
    print(f"Profiles per task count: {NUM_PROFILES_PER_TASK_COUNT}")
    print(f"Simulation repeats per profile: {SIMULATION_REPEATS}")
    print(f"Period range: {MIN_PERIOD}-{MAX_PERIOD}ms")
    print(f"Utilization range: {MIN_UTILIZATION:.2f}-{MAX_UTILIZATION:.2f}")
    print(f"Base seed: {BASE_SEED}")
    print(f"Total experiments: {len(TASK_COUNTS) * NUM_PROFILES_PER_TASK_COUNT * SIMULATION_REPEATS}")
    
    # ========================================
    # EXPERIMENT EXECUTION
    # ========================================
    
    print(f"\nStarting estimation time analysis...")
    experiment_start_time = time.time()
    
    # Initialize results collection
    all_estimation_times = []
    estimation_times_by_task_count = {count: [] for count in TASK_COUNTS}
    
    # Process each task count
    for task_count in TASK_COUNTS:
        print(f"\n{'='*60}")
        print(f"Processing Task Count: {task_count}")
        print(f"{'='*60}")
        
        # Calculate simulation duration for this task count
        simulation_duration = MAX_PERIOD * task_count * SIMULATION_DURATION_MULTIPLIER
        
        # Create profile generator for this task count
        profile_generator = RandomProfileGenerator(
            min_period=MIN_PERIOD,
            max_period=MAX_PERIOD,
            minimum_utilization=MIN_UTILIZATION,
            maximum_utilization=MAX_UTILIZATION
        )
        
        # Generate random profiles for this task count
        generated_profile_names = profile_generator.generate_profiles(
            task_count=task_count,
            num_profiles=NUM_PROFILES_PER_TASK_COUNT,
            duration=simulation_duration,
            time_unit=TIME_UNIT,
            seed=BASE_SEED + task_count * 100,
            keyword=KEYWORD
        )
        
        if not generated_profile_names:
            print(f"Error: No profiles were generated for task count {task_count}")
            continue
        
        print(f"Generated {len(generated_profile_names)} profiles for task count {task_count}")
        
        # Process each profile for this task count
        for profile_idx, profile_name in enumerate(generated_profile_names, 1):
            print(f"\n  Profile {profile_idx}/{len(generated_profile_names)}: {profile_name}")
            
            # Create simulator instance for this profile
            profile_file = Path("random_profiles") / profile_name
            simulator = CauseEffectChainSimulator(str(profile_file))
            
            # Load and validate profile
            if not simulator.load_profile():
                print(f"    Failed to load profile {profile_name}. Skipping...")
                continue
                
            if not simulator.validate_profile():
                print(f"    Profile validation failed for {profile_name}. Skipping...")
                continue
            
            # Run simulations for this profile
            for run_num in range(1, SIMULATION_REPEATS + 1):
                try:
                    # Determine seed for this run
                    run_seed = BASE_SEED + task_count * 1000 + profile_idx * 100 + run_num
                    
                    # Show progress
                    print(f"    Run {run_num:2d}/{SIMULATION_REPEATS}", end="")
                    
                    # Set seed for this run
                    if run_seed is not None:
                        random.seed(run_seed)
                    
                    # Run the simulation
                    results = simulator.run(seed=run_seed)
                    
                    # Create estimator instance
                    estimator = CauseEffectChainEstimator()
                    
                    
                    estimation_result = estimator.estimate_cause_effect_chains(
                        results, 
                        simulation_id=f"task{task_count}_{profile_name}_run_{run_num:03d}"
                    )
                    
                    estimation_time = estimation_result.estimation_time * 1000  # Convert to milliseconds
                    
                    # Store results
                    result_data = {
                        'task_count': task_count,
                        'profile_id': profile_name,
                        'simulation_repeat': run_num,
                        'estimation_time': estimation_time
                    }
                    
                    all_estimation_times.append(result_data)
                    estimation_times_by_task_count[task_count].append(estimation_time)
                    
                    print(f" ✓ ({estimation_time:.1f}ms)")
                    
                except Exception as e:
                    print(f" ✗ (Error: {e})")
                    continue
            
            # Clean up memory after each profile
            gc.collect()
    
    # ========================================
    # RESULTS ANALYSIS AND VISUALIZATION
    # ========================================
    
    experiment_end_time = time.time()
    total_experiment_time = experiment_end_time - experiment_start_time
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETED")
    print(f"{'='*80}")
    print(f"Total experiment time: {total_experiment_time:.2f} seconds")
    print(f"Total estimation experiments: {len(all_estimation_times)}")
    
    # Print summary statistics
    print(f"\n=== Estimation Time Summary ===")
    for task_count in TASK_COUNTS:
        if estimation_times_by_task_count[task_count]:
            times = estimation_times_by_task_count[task_count]
            mean_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            print(f"Task Count {task_count:2d}: Mean={mean_time:.1f}ms, Std={std_time:.1f}ms, "
                  f"Min={min_time:.1f}ms, Max={max_time:.1f}ms, N={len(times)}")
    
    # Export results to CSV
    print(f"\n=== Exporting Results ===")
    export_estimation_time_results(all_estimation_times, "estimation_times.csv")
    
    # Generate whisker plot
    print(f"\n=== Generating Visualization ===")
    generate_whisker_plot(estimation_times_by_task_count, "estimation_time_whisker_plot.png", 
                         show_title=False, font_size=24)
    
    print(f"\nResults saved in:")
    print(f"  - estimation_time_results/estimation_times.csv")
    print(f"  - estimation_time_results/estimation_time_whisker_plot.png")


if __name__ == "__main__":
    main()
