#!/usr/bin/env python3
"""
Cause-Effect Chain Semantic Simulator

This program simulates cause-effect chain semantics from a given YAML profile.
It analyzes task timing, dependencies, and execution patterns.
"""

import yaml
import sys
import random
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TaskProfile:
    """Represents a single task's semantic profile"""
    phase: int      # Start time offset (ms)
    period: int     # Recurrence period (ms)
    wcet: int       # Worst case execution time (ms)


@dataclass
class SimulationConfig:
    """Configuration for the simulation"""
    duration: int
    time_unit: str


@dataclass
class TaskEvent:
    """Represents a single event for a task"""
    task_name: str
    event_type: str  # 'release', 'read-event', 'write-event'
    timestamp: int


@dataclass
class TaskSimulationResult:
    """Contains simulation results for a single task"""
    task_name: str
    events: List[TaskEvent]
    release_times: List[int]
    read_event_times: List[int]
    write_event_times: List[int]


class CauseEffectChainSimulator:
    """Main simulator class for cause-effect chain analysis"""
    
    def __init__(self, profile_path: str):
        self.profile_path = profile_path
        self.config = None
        self.tasks = {}
        self.task_count = 0
        
    def load_profile(self) -> bool:
        """Load and parse the YAML profile file"""
        try:
            with open(self.profile_path, 'r') as file:
                profile_data = yaml.safe_load(file)
            
            if not profile_data:
                print(f"Error: Empty or invalid YAML file: {self.profile_path}")
                return False
                
            # Parse simulation configuration
            if 'simulation' in profile_data:
                sim_config = profile_data['simulation']
                self.config = SimulationConfig(
                    duration=sim_config.get('duration', 1000),
                    time_unit=sim_config.get('time_unit', 'ms')
                )
            
            # Parse task profiles
            if 'tasks' in profile_data:
                tasks_data = profile_data['tasks']
                self.task_count = tasks_data.get('count', 0)
                
                # Parse individual task profiles
                for task_key, task_data in tasks_data.items():
                    if task_key == 'count':
                        continue
                    
                    if isinstance(task_data, dict) and all(k in task_data for k in ['phase', 'period', 'wcet']):
                        self.tasks[task_key] = TaskProfile(
                            phase=task_data['phase'],
                            period=task_data['period'],
                            wcet=task_data['wcet']
                        )
            
            # print(f"Successfully loaded profile: {self.profile_path}")
            # print(f"Task count: {self.task_count}")
            # print(f"Simulation duration: {self.config.duration if self.config else 'Not specified'} {self.config.time_unit if self.config else ''}")
            
            return True
            
        except FileNotFoundError:
            print(f"Error: Profile file not found: {self.profile_path}")
            return False
        except yaml.YAMLError as e:
            print(f"Error: Invalid YAML format: {e}")
            return False
        except Exception as e:
            print(f"Error: Unexpected error loading profile: {e}")
            return False
    
    def validate_profile(self) -> bool:
        """Validate the loaded profile for consistency"""
        if not self.tasks:
            print("Error: No valid tasks found in profile")
            return False
            
        if len(self.tasks) != self.task_count:
            print(f"Warning: Task count mismatch. Expected: {self.task_count}, Found: {len(self.tasks)}")
        
        # Validate timing constraints
        for task_name, task in self.tasks.items():
            if task.phase < 0:
                print(f"Error: Negative phase for task {task_name}: {task.phase}")
                return False
            if task.period <= 0:
                print(f"Error: Invalid period for task {task_name}: {task.period}")
                return False
            if task.wcet <= 0:
                print(f"Error: Invalid WCET for task {task_name}: {task.wcet}")
                return False
            if task.wcet > task.period:
                print(f"Warning: WCET exceeds period for task {task_name}: {task.wcet} > {task.period}")
        
        # print("Profile validation completed successfully")
        return True
    
    def print_profile_summary(self):
        """Print a summary of the loaded profile"""
        print("\n=== Profile Summary ===")
        print(f"Total tasks: {len(self.tasks)}")
        print(f"Simulation duration: {self.config.duration if self.config else 'N/A'} {self.config.time_unit if self.config else ''}")
        print("\nTask Details:")
        print("-" * 50)
        
        for task_name, task in self.tasks.items():
            print(f"{task_name:10} | Phase: {task.phase:3}ms | Period: {task.period:3}ms | WCET: {task.wcet:2}ms")
        
        print("-" * 50)
    
    def run(self, seed: int = None, run_wcet: bool = False) -> Dict[str, TaskSimulationResult]:
        """
        Run the simulation and generate timestamp data for all tasks
        
        Args:
            seed: Random seed for reproducible results (optional)
            
        Returns:
            Dictionary mapping task names to their simulation results
        """
        if seed is not None:
            random.seed(seed)
            # print(f"Using random seed: {seed}")
        
        if not self.config:
            # print("Error: No simulation configuration loaded")
            return {}
        
        # print(f"\nStarting simulation for {self.config.duration} {self.config.time_unit}...")
        
        simulation_results = {}
        
        # Simulate each task independently
        for task_name, task_profile in self.tasks.items():
            # print(f"Simulating task: {task_name}")
            
            # Initialize result structure for this task
            task_result = TaskSimulationResult(
                task_name=task_name,
                events=[],
                release_times=[],
                read_event_times=[],
                write_event_times=[]
            )
            
            # Start simulation from a random time between 0 and phase
            current_time = task_profile.phase
            
            # Continue simulation until we exceed the simulation duration
            while current_time  < self.config.duration:
                # Mark current time as release event
                release_time = current_time
                task_result.release_times.append(release_time)
                
                # Create release event
                release_event = TaskEvent(
                    timestamp=release_time,
                    event_type='release',
                    task_name=task_name
                )
                task_result.events.append(release_event)
                
                # Pick random execution time between (0, WCET] in uniform distribution
                if run_wcet:
                    execution_time = task_profile.wcet
                else:
                    execution_time = random.uniform(task_profile.wcet*0.9, task_profile.wcet)
                
                # Pick read-event time between [current_time, current_time + period - execution_time]
                # Ensure read-event happens before the next period
                max_read_time = current_time + task_profile.period - execution_time
                if max_read_time > current_time:
                    read_event_time = random.uniform(current_time, max_read_time)
                else:
                    # If execution time is too close to period, read immediately
                    read_event_time = current_time
                
                if read_event_time > self.config.duration:
                    break

                # Create read event
                read_event = TaskEvent(
                    timestamp=int(read_event_time),
                    event_type='read-event',
                    task_name=task_name
                )
                task_result.read_event_times.append(int(read_event_time))
                task_result.events.append(read_event)
                
                # Set write-event time as read-event + execution time
                write_event_time = read_event_time + execution_time
                
                if write_event_time > self.config.duration:
                    break
                
                # Create write event
                write_event = TaskEvent(
                    timestamp=int(write_event_time),
                    event_type='write-event',
                    task_name=task_name
                )
                task_result.write_event_times.append(int(write_event_time))
                task_result.events.append(write_event)
                
                # Move to next period
                current_time += task_profile.period
            
            # Sort events by timestamp for chronological order
            task_result.events.sort(key=lambda x: x.timestamp)
            
            # Store results for this task
            simulation_results[task_name] = task_result
            
            # print(f"  Generated {len(task_result.events)} events for {task_name}")
            # print(f"  Release events: {len(task_result.release_times)}")
            # print(f"  Read events: {len(task_result.read_event_times)}")
            # print(f"  Write events: {len(task_result.write_event_times)}")
        
        # print(f"\nSimulation completed! Generated data for {len(simulation_results)} tasks.")
        return simulation_results
    
    def print_simulation_summary(self, results: Dict[str, TaskSimulationResult]):
        """Print a summary of the simulation results"""
        print("\n=== Simulation Results Summary ===")
        
        for task_name, result in results.items():
            print(f"\nTask: {task_name}")
            print(f"  Total events: {len(result.events)}")
            print(f"  Release events: {len(result.release_times)}")
            print(f"  Read events: {len(result.read_event_times)}")
            print(f"  Write events: {len(result.write_event_times)}")
            
            if result.events:
                first_event = min(result.events, key=lambda x: x.timestamp)
                last_event = max(result.events, key=lambda x: x.timestamp)
                print(f"  Time range: {first_event.timestamp}ms - {last_event.timestamp}ms")
                
                # Show first few events as example
                print("  Sample events:")
                for i, event in enumerate(result.events[:5]):
                    print(f"    {event.timestamp:4}ms: {event.event_type}")
                if len(result.events) > 5:
                    print(f"    ... and {len(result.events) - 5} more events")

    def export_simulation_trace(self, results: Dict[str, TaskSimulationResult], 
                               profile_name: str, 
                               run_id: int) -> None:
        """
        Export simulation trace to CSV file (fixed pattern, compressed format)
        
        Args:
            results: Dictionary mapping task names to their simulation results
            profile_name: Name of the profile for the output filename
            run_id: Run ID for the output filename
        """
        import csv
        from pathlib import Path
        
        # Create output directory if it doesn't exist
        output_dir = Path("simulation_traces")
        output_dir.mkdir(exist_ok=True)
        
        # Create CSV filename with profile name and run ID
        csv_filename = output_dir / f"simulation_trace_{profile_name}_run_{run_id:03d}.csv"
        
        try:
            with open(csv_filename, 'w', newline='') as csvfile:
                # Get all task names and sort them for consistent column order
                task_names = sorted(results.keys())
                
                # Write header row
                header = ['Event'] + task_names
                writer = csv.writer(csvfile)
                writer.writerow(header)
                
                # Find the maximum number of events across all tasks
                max_events = 0
                for task_result in results.values():
                    max_events = max(max_events, len(task_result.release_times))
                
                # Iterate through each event index (0, 1, 2, ...)
                for event_index in range(max_events):
                    # Check if any task has a release event at this index
                    has_release = any(event_index < len(results[task_name].release_times) 
                                    for task_name in task_names)
                    
                    # Check if any task has a read event at this index
                    has_read = any(event_index < len(results[task_name].read_event_times) 
                                 for task_name in task_names)
                    
                    # Check if any task has a write event at this index
                    has_write = any(event_index < len(results[task_name].write_event_times) 
                                  for task_name in task_names)
                    
                    # Write Release row if any task has this release event
                    if has_release:
                        release_row = ['Release']
                        for task_name in task_names:
                            if event_index < len(results[task_name].release_times):
                                release_row.append(str(int(results[task_name].release_times[event_index])))
                            else:
                                release_row.append('')
                        writer.writerow(release_row)
                    
                    # Write Read-event row if any task has this read event
                    if has_read:
                        read_row = ['Read-event']
                        for task_name in task_names:
                            if event_index < len(results[task_name].read_event_times):
                                read_row.append(str(int(results[task_name].read_event_times[event_index])))
                            else:
                                read_row.append('')
                        writer.writerow(read_row)
                    
                    # Write Write-event row if any task has this write event
                    if has_write:
                        write_row = ['Write-event']
                        for task_name in task_names:
                            if event_index < len(results[task_name].write_event_times):
                                write_row.append(str(int(results[task_name].write_event_times[event_index])))
                            else:
                                write_row.append('')
                        writer.writerow(write_row)
            
            # print(f"Simulation trace exported to: {csv_filename}")
            
        except Exception as e:
            print(f"Error exporting simulation trace: {e}")


def main():
    """Main function to run the cause-effect chain simulator"""
    print("Cause-Effect Chain Semantic Simulator")
    print("=" * 50)
    
    # Default profile path
    profile_path = "profile1.yaml"
    
    # Check if profile path is provided as command line argument
    if len(sys.argv) > 1:
        profile_path = sys.argv[1]
    
    # Check if profile file exists
    if not Path(profile_path).exists():
        print(f"Error: Profile file not found: {profile_path}")
        print("Usage: python cause-effect-chain-semantic-simulator.py [profile_path]")
        sys.exit(1)
    
    # Create simulator instance
    simulator = CauseEffectChainSimulator(profile_path)
    
    # Load and parse the profile
    if not simulator.load_profile():
        print("Failed to load profile. Exiting.")
        sys.exit(1)
    
    # Validate the profile
    if not simulator.validate_profile():
        print("Profile validation failed. Exiting.")
        sys.exit(1)
    
    # Print profile summary
    simulator.print_profile_summary()
    
    # Run the simulation
    simulation_results = simulator.run(seed=42)  # Use seed 42 for reproducible results
    
    # Print simulation summary
    simulator.print_simulation_summary(simulation_results)
    
    # Export simulation results
    # simulator.export_simulation_results(simulation_results) # This line is removed
    
    print("\nSimulation completed successfully!")
    


if __name__ == "__main__":
    main()
