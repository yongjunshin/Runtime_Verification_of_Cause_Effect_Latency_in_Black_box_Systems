#!/usr/bin/env python3
"""
RQ2-4 Experiment: WCET Utilization Impact Analysis

This experiment compares three different WCET utilization patterns:
- Profile 1: Heterogeneous WCET (random WCET < period for each task)
- Profile 2: High utilization (WCET/period = 0.9 for all tasks)
- Profile 3: Low utilization (WCET/period = 0.1 for all tasks)

All profiles have:
- Same number of tasks (4 tasks)
- Same period (100ms)
- Random phases
- Single simulation run (no repeats)
"""

import sys
import time
import csv
import random
import yaml
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
sys.path.append(current_dir)  # Add current directory for RQ2-4 specific modules

# Import the simulator, visualizer, and estimator
from util.cause_effect_chain_semantic_simulator import CauseEffectChainSimulator, TaskSimulationResult
from cause_effect_chain_visualizer_rq2_4 import CauseEffectChainVisualizer, GanttChartConfig
from util.cause_effect_chain_estimator_new import CauseEffectChainEstimator
from util.estimation_error_visualizer import EstimationErrorVisualizer, ScatterPlotConfig


def create_common_phases(task_count: int = 4, period: int = 100) -> List[float]:
    """
    Create common phases for all profiles to ensure fair comparison
    """
    random.seed(42)  # Fixed seed for reproducibility
    phases = []
    for i in range(task_count):
        # Random phase between 0 and period
        phase = random.uniform(0, period)
        phases.append(phase)
    return phases


def create_profile_1_heterogeneous(task_count: int = 4, period: int = 100, duration: int = 1000, phases: List[float] = None) -> Dict[str, Any]:
    """
    Create Profile 1: Heterogeneous WCET (random WCET < period for each task)
    """
    if phases is None:
        phases = create_common_phases(task_count, period)
    
    random.seed(42)  # Fixed seed for reproducibility
    
    tasks = {}
    for i in range(task_count):
        # Use common phase
        phase = phases[i]
        
        # Random WCET between 10% and 90% of period (heterogeneous)
        wcet = random.uniform(0.1 * period, 0.9 * period)
        
        tasks[f"task_{i+1}"] = {
            "phase": phase,
            "period": period,
            "wcet": wcet
        }
    
    profile = {
        "tasks": {
            "count": task_count,
            **tasks
        },
        "simulation": {
            "duration": duration,
            "time_unit": "ms"
        }
    }
    
    return profile


def create_profile_2_high_utilization(task_count: int = 4, period: int = 100, utilization: float = 0.9, duration: int = 1000, phases: List[float] = None) -> Dict[str, Any]:
    """
    Create Profile 2: High utilization (WCET/period = 0.9 for all tasks)
    """
    if phases is None:
        phases = create_common_phases(task_count, period)
    
    tasks = {}
    for i in range(task_count):
        # Use common phase
        phase = phases[i]
        
        # WCET = utilization * period
        wcet = utilization * period
        
        tasks[f"task_{i+1}"] = {
            "phase": phase,
            "period": period,
            "wcet": wcet
        }
    
    profile = {
        "tasks": {
            "count": task_count,
            **tasks
        },
        "simulation": {
            "duration": duration,
            "time_unit": "ms"
        }
    }
    
    return profile


def create_profile_3_low_utilization(task_count: int = 4, period: int = 100, utilization: float = 0.1, duration: int = 1000, phases: List[float] = None) -> Dict[str, Any]:
    """
    Create Profile 3: Low utilization (WCET/period = 0.1 for all tasks)
    """
    if phases is None:
        phases = create_common_phases(task_count, period)
    
    tasks = {}
    for i in range(task_count):
        # Use common phase
        phase = phases[i]
        
        # WCET = utilization * period
        wcet = utilization * period
        
        tasks[f"task_{i+1}"] = {
            "phase": phase,
            "period": period,
            "wcet": wcet
        }
    
    profile = {
        "tasks": {
            "count": task_count,
            **tasks
        },
        "simulation": {
            "duration": duration,
            "time_unit": "ms"
        }
    }
    
    return profile


def save_profile(profile: Dict[str, Any], filename: str) -> None:
    """
    Save profile to YAML file
    """
    profiles_dir = Path("random_profiles")
    profiles_dir.mkdir(exist_ok=True)
    
    file_path = profiles_dir / f"{filename}.yaml"
    
    with open(file_path, 'w') as f:
        yaml.dump(profile, f, default_flow_style=False, sort_keys=False)
    
    print(f"  ✓ Profile saved: {file_path}")


def export_evaluation_results(results_data: List[Dict[str, Any]], profile_name: str) -> None:
    """
    Export evaluation results to CSV file
    """
    # Create output directory if it doesn't exist
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Create CSV filename with profile name
    csv_filename = output_dir / f"evaluation_results_{profile_name}.csv"
    
    # Define CSV headers
    headers = ['repeat', 'oracle_chain_length', 'estimated_chain_length', 'error', 'error_percentage']
    
    # Write CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results_data)
    
    print(f"  ✓ Evaluation results exported: {csv_filename}")


def main():
    """Main experiment function"""
    
    # ========================================
    # EXPERIMENT CONFIGURATION
    # ========================================
    
    TASK_COUNT = 4
    PERIOD = 100  # ms
    SIMULATION_DURATION = 1000  # ms
    SIMULATION_REPEATS = 1  # Single run per profile
    BASE_SEED = 222
    
    print("=" * 80)
    print("RQ2-4 EXPERIMENT: WCET Utilization Impact Analysis")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Task count: {TASK_COUNT}")
    print(f"  Period: {PERIOD}ms")
    print(f"  Simulation duration: {SIMULATION_DURATION}ms")
    print(f"  Repeats per profile: {SIMULATION_REPEATS}")
    print(f"  Base seed: {BASE_SEED}")
    print()
    
    # ========================================
    # CREATE PROFILES
    # ========================================
    
    print("Creating experimental profiles...")
    
    # Create common phases for all profiles
    common_phases = create_common_phases(TASK_COUNT, PERIOD)
    print(f"  Common phases: {[f'{p:.1f}ms' for p in common_phases]}")
    
    # Profile 1: Heterogeneous WCET
    print("  Creating Profile 1: Heterogeneous WCET")
    profile1 = create_profile_1_heterogeneous(TASK_COUNT, PERIOD, SIMULATION_DURATION, common_phases)
    save_profile(profile1, "RQ2_4_heterogeneous_4tasks")
    
    # Profile 2: High utilization (0.9)
    print("  Creating Profile 2: High utilization (0.9)")
    profile2 = create_profile_2_high_utilization(TASK_COUNT, PERIOD, 0.9, SIMULATION_DURATION, common_phases)
    save_profile(profile2, "RQ2_4_high_util_4tasks")
    
    # Profile 3: Low utilization (0.1)
    print("  Creating Profile 3: Low utilization (0.1)")
    profile3 = create_profile_3_low_utilization(TASK_COUNT, PERIOD, 0.1, SIMULATION_DURATION, common_phases)
    save_profile(profile3, "RQ2_4_low_util_4tasks")
    
    print()
    
    # ========================================
    # EXPERIMENT EXECUTION
    # ========================================
    
    print("Starting experiment execution...")
    experiment_start_time = time.time()
    
    profile_names = [
        "RQ2_4_heterogeneous_4tasks",
        "RQ2_4_high_util_4tasks", 
        "RQ2_4_low_util_4tasks"
    ]
    
    print(f"Processing {len(profile_names)} profiles")
    print()
    
    # Process each profile
    for profile_idx, profile_name in enumerate(profile_names, 1):
        print(f"Profile {profile_idx}/{len(profile_names)}: {profile_name}")
        
        # Create simulator instance for this profile
        profile_file = Path("random_profiles") / f"{profile_name}.yaml"
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
        
        # Run single simulation
        for run_num in range(1, SIMULATION_REPEATS + 1):
            try:
                # Determine seed for this run
                run_seed = BASE_SEED
                
                # Show progress
                print(f"  Run {run_num:2d}/{SIMULATION_REPEATS}", end="")
                
                # Set seed for this run
                if run_seed is not None:
                    random.seed(run_seed)
                
                # Run the simulation
                results = simulator.run(seed=run_seed)
                
                # Export simulation trace
                simulator.export_simulation_trace(results, profile_name, run_num)
                
                # Create estimator instance
                estimator = CauseEffectChainEstimator()
                
                # Perform chain estimation
                try:
                    estimation_result = estimator.estimate_cause_effect_chains(
                        results, 
                        simulation_id=f"{profile_name}_run_{run_num:03d}"
                    )
                    
                    # Export chain comparison results
                    estimator.export_chain_comparison(estimation_result, profile_name, run_num)
                    
                except Exception as e:
                    print(f" ✗ (estimation failed)")
                    estimation_result = None
                
                # Calculate evaluation metrics
                oracle_chain_length = None
                estimated_chain_length = None
                error = None
                error_percentage = None
                
                if estimation_result and estimation_result.oracle_chains and estimation_result.oracle_chains[0].chain_length is not None:
                    oracle_chain_length = estimation_result.oracle_chains[0].chain_length
                
                if estimation_result and estimation_result.estimated_chains and estimation_result.estimated_chains[0].chain_length is not None:
                    estimated_chain_length = estimation_result.estimated_chains[0].chain_length
                
                if oracle_chain_length is not None and estimated_chain_length is not None:
                    error = estimated_chain_length - oracle_chain_length
                    error_percentage = (error / oracle_chain_length) * 100
                
                # Store evaluation results
                evaluation_results.append({
                    'repeat': run_num,
                    'oracle_chain_length': oracle_chain_length,
                    'estimated_chain_length': estimated_chain_length,
                    'error': error,
                    'error_percentage': error_percentage
                })
                
                if oracle_chain_length is not None and estimated_chain_length is not None:
                    print(f" ✓ (oracle: {oracle_chain_length:.1f}ms, estimated: {estimated_chain_length:.1f}ms, error: {error_percentage:.1f}%)")
                else:
                    print(f" ✓ (estimation data incomplete)")
                
                # Create visualizer and generate Gantt chart with custom x-range control
                try:
                    # Define different configurations for each profile
                    if "heterogeneous" in profile_name:
                        # Profile 1: Show title, medium font size
                        chart_config = GanttChartConfig(
                            title=f"Cause-Effect Chain Simulation - {profile_name}",
                            figure_width=7,
                            figure_height=7,
                            x_min=200,
                            x_max=1000,
                            x_tick_interval=100,
                            xlabel="Time (ms)",
                            show_title=False,
                            font_size=16
                        )
                        print(f"    → X-range: 200-1000ms, title: ON, font: 12pt")
                    elif "high_util" in profile_name:
                        # Profile 2: Hide title, large font size
                        chart_config = GanttChartConfig(
                            title=f"Cause-Effect Chain Simulation - {profile_name}",
                            figure_width=7,
                            figure_height=7,
                            x_min=200,
                            x_max=1000,
                            x_tick_interval=100,
                            xlabel="Time (ms)",
                            show_title=False,
                            font_size=16
                        )
                        print(f"    → X-range: 200-1000ms, title: OFF, font: 14pt")
                    elif "low_util" in profile_name:
                        # Profile 3: Show title, small font size
                        chart_config = GanttChartConfig(
                            title=f"Cause-Effect Chain Simulation - {profile_name}",
                            figure_width=7,
                            figure_height=7,
                            x_min=200,
                            x_max=1000,
                            x_tick_interval=100,
                            xlabel="Time (ms)",
                            show_title=False,
                            font_size=16
                        )
                        print(f"    → X-range: 200-1000ms, title: ON, font: 10pt")
                    else:
                        # Default configuration
                        chart_config = GanttChartConfig(
                            title=f"Cause-Effect Chain Simulation - {profile_name}",
                            figure_width=14,
                            figure_height=10,
                            show_title=True,
                            font_size=16
                        )
                        print(f"    → X-range: Auto, title: ON, font: 12pt")
                    
                    visualizer = CauseEffectChainVisualizer(chart_config)
                    
                    # Generate and save Gantt chart with both simulation and estimation results
                    output_dir = "visualization_results/"
                    output_file = output_dir + f"gantt_chart_{profile_name}_run_{run_num:03d}.png"
                    visualizer.generate_and_save_gantt_chart(results, output_file, estimation_result)
                    print(f"    ✓ Chart saved: {output_file}")
                    
                    # Clean up matplotlib memory
                    plt.close('all')
                    visualizer.clear_last_figure()
                    
                except Exception as e:
                    print(f" ✗ (visualization failed: {e})")
                
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
    print(f"Profiles processed: {len(profile_names)}")
    print(f"Total simulations: {len(profile_names) * SIMULATION_REPEATS}")
    print(f"Results saved in:")
    print(f"  - random_profiles/: Generated profile files")
    print(f"  - simulation_traces/: Simulation trace data")
    print(f"  - estimation_results/: Chain comparison data")
    print(f"  - evaluation_results/: Error analysis data")
    print(f"  - evaluation_results_plots/: Error analysis plots")
    print(f"  - visualization_results/: Gantt charts")


if __name__ == "__main__":
    main()
