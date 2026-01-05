#!/usr/bin/env python3
"""
Main Experiment Controller for Cause-Effect Chain Semantic Simulation

This script controls the overall experiment by generating random profiles and
running multiple simulations with each profile to gather statistical data.
"""

import sys
import time
import csv
import random
from pathlib import Path
from typing import Dict, List, Any
import statistics
import matplotlib.pyplot as plt
import gc
import os

# Add parent directory to path to access util modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import the simulator, visualizer, and random profile generator
from util.cause_effect_chain_semantic_simulator import CauseEffectChainSimulator, TaskSimulationResult
from util.cause_effect_chain_visualizer import CauseEffectChainVisualizer, GanttChartConfig
from util.cause_effect_chain_estimator_new import CauseEffectChainEstimator
from util.estimation_error_visualizer import EstimationErrorVisualizer, ScatterPlotConfig
from util.random_profile_generator import RandomProfileGenerator



def export_evaluation_results(results_data: List[Dict[str, Any]], profile_name: str) -> None:
    """
    Export evaluation results to CSV file
    
    Args:
        results_data: List of dictionaries containing evaluation results
        profile_name: Name of the profile for the output filename
    """
    # Create output directory if it doesn't exist
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Create CSV filename with profile name
    csv_filename = output_dir / f"evaluation_results_{profile_name}.csv"
    
    # Define CSV headers
    headers = ['repeat', 'oracle_chain_length', 'estimated_chain_length', 'error', 'error_percentage']
    
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            
            # Write header
            writer.writeheader()
            
            # Write data rows
            for result in results_data:
                writer.writerow(result)

    except Exception as e:
        print(f"Error exporting evaluation results: {e}")


def main():
    """Main experiment controller"""
    print("Cause-Effect Chain Semantic Simulation - Experiment Controller")
    print("=" * 80)
    
    # ========================================
    # EXPERIMENT PARAMETER SETTINGS
    # ========================================
    
    Ns=[2, 10]
    Ts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    Us = [0.1, 0.9]

    for N in Ns:
        for T in Ts:
            for U in Us:
        # Random profile generator parameters
                MIN_PERIOD = 20              # Minimum period in ms
                MAX_PERIOD = 100             # Maximum period in ms
                MIN_UTILIZATION = U        # Minimum WCET/period ratio
                MAX_UTILIZATION = U        # Maximum WCET/period ratio
                # KEYWORD = None
                KEYWORD = "Homo"
                POSTFIX = f"{T}_{U}"
                
                # Profile generation parameters
                TASK_COUNT = N               # Number of tasks in each chain {2, 4, 6, 8, 10, 12, 14, 16, 18, 20}
                NUM_PROFILES = 20            # Number of random profiles to generate
                SIMULATION_DURATION = MAX_PERIOD * (TASK_COUNT * 3 + 3)  # Simulation duration in ms (3T + 3\sumT)
                TIME_UNIT = "ms"             # Time unit
                
                # Simulation parameters
                SIMULATION_REPEATS = 5       # Number of simulation runs per profile
                BASE_SEED = 222               # Base random seed for reproducibility
                
                # ========================================
                # PRINT EXPERIMENT CONFIGURATION
                # ========================================
                
                print("=== Experiment Configuration ===")
                print(f"Profile Generation:")
                print(f"  Task count: {TASK_COUNT}")
                print(f"  Number of profiles: {NUM_PROFILES}")
                print(f"  Period range: {MIN_PERIOD}-{MAX_PERIOD}ms")
                print(f"  Utilization range: {MIN_UTILIZATION:.2f}-{MAX_UTILIZATION:.2f}")
                print(f"  Simulation duration: {SIMULATION_DURATION}{TIME_UNIT}")
                print(f"\nSimulation Parameters:")
                print(f"  Repeats per profile: {SIMULATION_REPEATS}")
                print(f"  Base seed: {BASE_SEED}")
                
                # ========================================
                # GENERATE RANDOM PROFILES
                # ========================================
                
                print(f"\nGenerating random profiles...")
                profile_generator = RandomProfileGenerator(
                    min_period=MIN_PERIOD,
                    max_period=MAX_PERIOD,
                    minimum_utilization=MIN_UTILIZATION,
                    maximum_utilization=MAX_UTILIZATION
                )
                
                # Generate random profiles
                generated_profile_names = profile_generator.generate_profiles(
                    task_count=TASK_COUNT,
                    num_profiles=NUM_PROFILES,
                    duration=SIMULATION_DURATION,
                    time_unit=TIME_UNIT,
                    seed=BASE_SEED,
                    keyword=KEYWORD,
                    postfix=POSTFIX
                )
                
                # ========================================
                # EXPERIMENT EXECUTION
                # ========================================
                
                print(f"\nStarting experiment execution...")
                experiment_start_time = time.time()
                
                
                
                # Use the generated profile names directly
                if not generated_profile_names:
                    print("Error: No profiles were generated")
                    sys.exit(1)
                
                print(f"Processing {len(generated_profile_names)} generated profiles for chains with {TASK_COUNT} tasks")
                
                # Process each profile
                for profile_idx, profile_name in enumerate(generated_profile_names, 1):
                    print(f"\nProfile {profile_idx}/{len(generated_profile_names)}: {profile_name}")
                    
                    # Create simulator instance for this profile
                    profile_file = Path("random_profiles") / profile_name
                    simulator = CauseEffectChainSimulator(str(profile_file))
                    
                    # Load and validate profile
                    if not simulator.load_profile():
                        print(f"Failed to load profile {profile_name}. Skipping...")
                        continue
                        
                    if not simulator.validate_profile():
                        print(f"Profile validation failed for {profile_name}. Skipping...")
                        continue
                    
                    # Initialize results collection for this profile
                    evaluation_results = []
                    
                    
                    for run_num in range(1, SIMULATION_REPEATS + 1):
                        try:
                            # Determine seed for this run
                            run_seed = BASE_SEED + profile_idx * 1000 + run_num
                            
                            # Show progress
                            print(f"  Run {run_num:2d}/{SIMULATION_REPEATS}", end="")
                            
                            # Set seed for this run
                            if run_seed is not None:
                                random.seed(run_seed)
                            
                            # Run the simulation
                            if KEYWORD == "Homo":
                                results = simulator.run(seed=run_seed, run_wcet=True)
                            else:
                                results = simulator.run(seed=run_seed)
                            
                            # Export simulation trace
                            simulator.export_simulation_trace(results, profile_name, run_num)
                            

                            # Create estimator instance
                            estimator = CauseEffectChainEstimator()
                            
                            try:
                                estimation_result = estimator.estimate_cause_effect_chains(
                                    results, 
                                    simulation_id=f"{profile_name}_run_{run_num:03d}"
                                )
                                
                                # Export chain comparison results
                                estimator.export_chain_comparison(estimation_result, profile_name, run_num)
                                
                                # Collect evaluation data
                                oracle_chain_length = None
                                estimated_chain_length = None
                                error = None
                                error_percentage = None
                                
                                if estimation_result.oracle_chains and estimation_result.oracle_chains[0].chain_length is not None:
                                    oracle_chain_length = estimation_result.oracle_chains[0].chain_length
                                
                                if estimation_result.estimated_chains and estimation_result.estimated_chains[0].chain_length is not None:
                                    estimated_chain_length = estimation_result.estimated_chains[0].chain_length
                                
                                if oracle_chain_length is not None and estimated_chain_length is not None:
                                    error = estimated_chain_length - oracle_chain_length
                                    error_percentage = (error / oracle_chain_length) * 100

                                evaluation_results.append({
                                    'repeat': run_num,
                                    'oracle_chain_length': oracle_chain_length,
                                    'estimated_chain_length': estimated_chain_length,
                                    'error': error,
                                    'error_percentage': error_percentage
                                })
                                
                                    
                            except Exception as e:
                                print(f" ✗ (estimation failed)")
                                estimation_result = None
                                
                                # Add failed run to results
                                evaluation_results.append({
                                    'repeat': run_num,
                                    'oracle_chain_length': None,
                                    'estimated_chain_length': None,
                                    'error': None,
                                    'error_percentage': None
                                })
                            
                            
                            # Create visualizer and generate Gantt chart
                            try:
                                if profile_idx == 1 and run_num == 1:
                                    chart_config = GanttChartConfig(
                                        title=f"Cause-Effect Chain Simulation #{run_num} - {profile_name}",
                                        figure_width=14,
                                        figure_height=10
                                    )
                                    visualizer = CauseEffectChainVisualizer(chart_config)
                                    
                                    # Generate and save Gantt chart with both simulation and estimation results
                                    output_dir = "visualization_results/"
                                    output_file = output_dir + f"gantt_chart_{profile_name}_run_{run_num:03d}.png"
                                    visualizer.generate_and_save_gantt_chart(results, output_file, estimation_result)
                                    
                                    
                                    # Clean up matplotlib memory
                                    plt.close('all')
                                    visualizer.clear_last_figure()
                                    
                            except Exception as e:
                                print(f" ✗ (visualization failed)")
                            
                            # Add newline after each run
                            print()
                            
                        except Exception as e:
                            print(f"Error in simulation #{run_num}: {e}")
                            # Clean up matplotlib memory even on error
                            plt.close('all')
                            continue
                    
                    # Export evaluation results for this profile
                    export_evaluation_results(evaluation_results, profile_name)
                    
                    # Generate estimation error visualizations for this profile
                    try:
                        error_visualizer = EstimationErrorVisualizer()
                        error_visualizer.generate_and_save_plots(evaluation_results, profile_name, "evaluation_results_plots")
                        
                        # Clean up matplotlib memory
                        plt.close('all')
                        error_visualizer.clear_last_figure()
                            
                    except Exception as e:
                        print(f"Warning: Error visualization failed for {profile_name}")
                        # Clean up matplotlib memory even on error
                        plt.close('all')
                    
                    # Clean up memory after processing each profile
                    gc.collect()
                    plt.close('all')
                
                # ========================================
                # EXPERIMENT COMPLETION
                # ========================================
                
                experiment_end_time = time.time()
                total_experiment_time = experiment_end_time - experiment_start_time
                
                print(f"\n{'='*80}")
                print(f"EXPERIMENT COMPLETED")
                print(f"{'='*80}")
                print(f"Total experiment time: {total_experiment_time:.2f} seconds")
                print(f"Profiles processed: {len(generated_profile_names)}")
                print(f"Total simulations: {len(generated_profile_names) * SIMULATION_REPEATS}")
                print(f"Results saved in:")
                print(f"  - random_profiles/: Generated profile files")
                print(f"  - simulation_traces/: Simulation trace data")
                print(f"  - estimation_results/: Chain comparison data")
                print(f"  - evaluation_results/: Error analysis data")
                print(f"  - evaluation_results_plots/: Error analysis plots")
                print(f"  - visualization_results/: Gantt charts")


if __name__ == "__main__":
    main()
