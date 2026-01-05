#!/usr/bin/env python3
"""
Cause-Effect Chain Estimator

This module analyzes simulation results to estimate cause-effect chain relationships
between tasks based on timing patterns and event sequences.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import copy
import time




class TaskEventPair:
    """Simple class for task event pairs"""
    def __init__(self, task_name: str, read_event: Any, write_event: Any, execution_time: int):
        self.task_name = task_name
        self.read_event = read_event
        self.write_event = write_event
        self.execution_time = execution_time


@dataclass
class CauseEffectChain:
    """Represents a complete cause-effect chain across multiple tasks"""
    chain_id: str
    task_sequence: List[TaskEventPair]  # List of (read-event, write-event) pairs
    total_chain_time: int               # Total time from first read to last write
    chain_confidence: float             # Overall confidence score for the chain
    chain_type: str                     # Type: 'oracle', 'estimated', 'partial'
    chain_length: Optional[int] = None  # Total length from z event to last write-event
    z_event: Optional[Any] = None  # Z event (first task's read-event)
    z_prime_event: Optional[Any] = None  # Z' event (last task's write-event)


@dataclass
class EstimationResult:
    """Contains only the essential estimation results"""
    simulation_id: str
    total_tasks: int
    oracle_chains: List[CauseEffectChain]  # Oracle-defined cause-effect chains
    estimated_chains: List[CauseEffectChain]  # Estimated cause-effect chains
    estimation_time: float  # Time taken for estimation in seconds



class CauseEffectChainEstimator:
    """Estimator class for analyzing cause-effect chain relationships"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the estimator
        
        Args:
            config: Configuration dictionary for estimation parameters (not used in simplified version)
        """
        self.config = config or {}
    
    def estimate_cause_effect_chains(self, simulation_results: Dict[str, Any], 
                                   simulation_id: str = "unknown") -> EstimationResult:
        """
        Main method to estimate cause-effect chains from simulation results
        
        Args:
            simulation_results: Results from the simulator
            simulation_id: Identifier for this simulation run
            
        Returns:
            EstimationResult containing all analysis results
        """
        if not simulation_results:
            raise ValueError("No simulation results provided")
        
        # print(f"Starting cause-effect chain estimation for simulation: {simulation_id}")
        
        # Extract task information
        task_names = list(simulation_results.keys())
        total_tasks = len(task_names)
        
        # Execute the two core estimation methods
        # print("  Executing oracle chain definition...")
        oracle_chains = self.oracle_chain_definition(simulation_results)
        
        # print("  Executing chain estimation...")
        estimation_start_time = time.time()
        estimated_chains = self.chain_estimation(simulation_results)
        estimation_end_time = time.time()
        estimation_time = estimation_end_time - estimation_start_time
        
        # Return both oracle and estimated chains
        # print(f"  Found {len(oracle_chains)} oracle chains and {len(estimated_chains)} estimated chains")
        
        # Create estimation result with only essential data
        estimation_result = EstimationResult(
            simulation_id=simulation_id,
            total_tasks=total_tasks,
            oracle_chains=oracle_chains,
            estimated_chains=estimated_chains,
            estimation_time=estimation_time
        )
        
        return estimation_result
    
    def oracle_chain_definition(self, simulation_results: Dict[str, Any]) -> List[CauseEffectChain]:
        """
        Define oracle cause-effect chains based on the simulation profile and results
        Using backward algorithm: from last task to first task
        
        Args:
            simulation_results: Results from the simulator
            
        Returns:
            List of oracle-defined cause-effect chains
        """
        if not simulation_results:
            return []
        
        # Get task names in order (assuming they are ordered in the profile)
        task_names = list(simulation_results.keys())
        if len(task_names) < 2:
            print("Need at least 2 tasks for cause-effect chain analysis")
            return []

        all_events = []

        for task_name, task_result in simulation_results.items():
            if hasattr(task_result, 'events') and task_result.events:
                all_events.extend(task_result.events)
        # print(all_events)

        # find maximum of x.timestamp for all events
        max_timestamp = max(all_events, key=lambda x: x.timestamp)
        # print(max_timestamp)
        # print("max:", max_timestamp)

        # print(f"  Analyzing {len(task_names)} tasks for oracle chain definition (backward algorithm)")
        
        # Initialize chain components
        task_sequence = []
        
        # Step 1: Find the last task's first-latest write-event as z'
        last_task = task_names[-1]
        last_task_result = simulation_results[last_task]

        

        
        if not hasattr(last_task_result, 'events') or not last_task_result.events:
            print(f"  No events found for last task: {last_task}")
            return []
        
        # Find write-events in last task
        last_write_events = [e for e in last_task_result.events if e.event_type == 'write-event']
        if not last_write_events:
            print(f"  No write events found for last task: {last_task}")
            return []
        
        # Sort write events by timestamp and get the first-latest (last write event)
        last_write_events.sort(key=lambda x: x.timestamp)
        z_prime_write_event = copy.deepcopy(last_write_events[-1])  # First-latest write event (z')
        z_prime_write_event.timestamp = max_timestamp.timestamp
        
        # For testing: add 10 to z_prime_write_event's timestamp
        # z_prime_write_event.timestamp += 1
        
        # print(f"  Last task {last_task}: z' event at {z_prime_write_event.timestamp}ms")
        
        # Step 2: Find the first-latest write- and read-events pair of the last task
        # Sort write events by timestamp and get the first-latest (last write event)
        last_write_events.sort(key=lambda x: x.timestamp)
        last_write_event = last_write_events[-1]  # First-latest write event
        
        # Find read-events in last task
        last_read_events = [e for e in last_task_result.events if e.event_type == 'read-event']
        if not last_read_events:
            print(f"  No read events found for last task: {last_task}")
            return []
        
        # Find read-event that comes before this write-event
        preceding_read_events = [e for e in last_read_events if e.timestamp <= last_write_event.timestamp]
        if not preceding_read_events:
            print(f"  No read event precedes write event for last task: {last_task}")
            return []
        
        last_read_event = max(preceding_read_events, key=lambda x: x.timestamp)  # Latest read event before write event
        
        # Create last task event pair
        last_execution_time = last_write_event.timestamp - last_read_event.timestamp
        last_pair = TaskEventPair(
            task_name=last_task,
            read_event=last_read_event,
            write_event=last_write_event,
            execution_time=last_execution_time
        )
        task_sequence.append(last_pair)
        
        # Set pivot time for previous task (read-event of last task)
        current_pivot_time = last_read_event.timestamp
        
        # print(f"  Last task {last_task} chain: read at {last_read_event.timestamp}ms, write at {last_write_event.timestamp}ms")
        
        # Step 3-5: Process remaining tasks backward using pivot times
        for i in range(len(task_names) - 2, -1, -1):  # Go backward from second-to-last to first task
            task_name = task_names[i]
            task_result = simulation_results[task_name]
            
            if not hasattr(task_result, 'events') or not task_result.events:
                print(f"  No events found for task: {task_name}")
                continue
            
            # Find write-events that come before the pivot time
            write_events = [e for e in task_result.events if e.event_type == 'write-event' and e.timestamp < current_pivot_time]
            if not write_events:
                print(f"  No write events found before pivot time {current_pivot_time}ms for task: {task_name}")
                continue
            
            # Find the latest write-event before pivot
            latest_write_event = max(write_events, key=lambda x: x.timestamp)
            
            # Find read-events that come before this write-event
            read_events = [e for e in task_result.events if e.event_type == 'read-event' and e.timestamp <= latest_write_event.timestamp]
            if not read_events:
                print(f"  No read events found before write event for task: {task_name}")
                continue
            
            # Find the latest read-event before the write-event
            latest_read_event = max(read_events, key=lambda x: x.timestamp)
            
            # Create task event pair
            execution_time = latest_write_event.timestamp - latest_read_event.timestamp
            task_pair = TaskEventPair(
                task_name=task_name,
                read_event=latest_read_event,
                write_event=latest_write_event,
                execution_time=execution_time
            )
            task_sequence.insert(0, task_pair)  # Insert at beginning to maintain order
            
            # Update pivot time for next previous task (read-event of current task)
            current_pivot_time = latest_read_event.timestamp
            
            # print(f"  Task {task_name}: read at {latest_read_event.timestamp}ms, write at {earliest_write_event.timestamp}ms")
        
        # Create the oracle chain
        if len(task_sequence) > 1:  # Need at least 2 tasks for a meaningful chain
            total_chain_time = task_sequence[-1].write_event.timestamp - task_sequence[0].read_event.timestamp
            
            # Calculate chain length: z' event - z event (first task read-event)
            chain_length = z_prime_write_event.timestamp - task_sequence[0].read_event.timestamp
            
            # z is the first task's read-event
            z_event = task_sequence[0].read_event
            
            oracle_chain = CauseEffectChain(
                chain_id=f"oracle_chain_{len(task_sequence)}_tasks",
                task_sequence=task_sequence,
                total_chain_time=total_chain_time,
                chain_confidence=1.0,  # Oracle chains have full confidence
                chain_type="oracle",
                chain_length=chain_length,
                z_event=z_event,  # Store z (first task's read-event) as the z_event
                z_prime_event=z_prime_write_event  # Store z' (last task's write-event) as the z_prime_event
            )
            
            # print(f"  Oracle chain created: {len(task_sequence)} tasks, total time: {total_chain_time}ms, chain length: {chain_length}ms")
            return [oracle_chain]
        else:
            # print("  Insufficient tasks for oracle chain creation")
            return []
    
    def chain_estimation(self, simulation_results: Dict[str, Any]) -> List[CauseEffectChain]:
        """
        Estimate cause-effect chains from simulation results using backward algorithm
        with second-latest write-event as pivot
        
        Args:
            simulation_results: Results from the simulator
            
        Returns:
            List of estimated cause-effect chains
        """
        if not simulation_results:
            return []
        
        # Get task names in order (assuming they are ordered in the profile)
        task_names = list(simulation_results.keys())
        if len(task_names) < 2:
            print("Need at least 2 tasks for cause-effect chain estimation")
            return []
        
        # print(f"  Estimating cause-effect chains for {len(task_names)} tasks (backward algorithm)")
        
        # Initialize chain components
        task_sequence = []
        
        # Find z' (largest timestamp across all events)
        all_events = []
        for task_name, task_result in simulation_results.items():
            if hasattr(task_result, 'events') and task_result.events:
                all_events.extend(task_result.events)
        
        if not all_events:
            print("  No events found across all tasks")
            return []
        
        # Find maximum timestamp across all events
        max_timestamp = max(all_events, key=lambda x: x.timestamp)
        
        # Step 1: Find the last task's first-latest write-event as z'
        last_task = task_names[-1]
        last_task_result = simulation_results[last_task]
        
        if not hasattr(last_task_result, 'events') or not last_task_result.events:
            print(f"  No events found for last task: {last_task}")
            return []
        
        # Find write-events in last task
        last_write_events = [e for e in last_task_result.events if e.event_type == 'write-event']
        if not last_write_events:
            print(f"  No write events found for last task: {last_task}")
            return []
        
        # Sort write events by timestamp and get the first-latest (last write event)
        last_write_events.sort(key=lambda x: x.timestamp)
        z_prime_write_event = copy.deepcopy(last_write_events[-1])  # First-latest write event (z')
        z_prime_write_event.timestamp = max_timestamp.timestamp
        
        # Step 2: Find the first-latest write- and read-events pair of the last task
        # Sort write events by timestamp and get the first-latest (last write event)
        last_write_events.sort(key=lambda x: x.timestamp)
        last_write_event = last_write_events[-1]  # First-latest write event
        
        # Find read-events in last task
        last_read_events = [e for e in last_task_result.events if e.event_type == 'read-event']
        if not last_read_events:
            print(f"  No read events found for last task: {last_task}")
            return []
        
        # Find read-event that comes before this write-event
        preceding_read_events = [e for e in last_read_events if e.timestamp <= last_write_event.timestamp]
        if not preceding_read_events:
            print(f"  No read event precedes write event for last task: {last_task}")
            return []
        
        last_read_event = max(preceding_read_events, key=lambda x: x.timestamp)  # Latest read event before write event
        
        # Create last task event pair
        last_execution_time = last_write_event.timestamp - last_read_event.timestamp
        last_pair = TaskEventPair(
            task_name=last_task,
            read_event=last_read_event,
            write_event=last_write_event,
            execution_time=last_execution_time
        )
        task_sequence.append(last_pair)
        
        # Set pivot time for previous task (second-latest write-event of last task)
        if len(last_write_events) >= 2:
            current_pivot_time = last_write_events[-2].timestamp  # Second-latest write event
        else:
            current_pivot_time = last_read_event.timestamp  # Fallback to read event if only one write event
        
        # Step 3-5: Process remaining tasks backward using second-latest write-event as pivot
        for i in range(len(task_names) - 2, -1, -1):  # Go backward from second-to-last to first task
            task_name = task_names[i]
            task_result = simulation_results[task_name]
            
            if not hasattr(task_result, 'events') or not task_result.events:
                print(f"  No events found for task: {task_name}")
                continue
            
            # Find write-events that come before the pivot time
            write_events = [e for e in task_result.events if e.event_type == 'write-event' and e.timestamp < current_pivot_time]
            if not write_events:
                print(f"  No write events found before pivot time {current_pivot_time}ms for task: {task_name}")
                continue
            
            # Find the latest write-event before pivot
            latest_write_event = max(write_events, key=lambda x: x.timestamp)
            
            # Find read-events that come before this write-event
            read_events = [e for e in task_result.events if e.event_type == 'read-event' and e.timestamp <= latest_write_event.timestamp]
            if not read_events:
                print(f"  No read events found before write event for task: {task_name}")
                continue
            
            # Find the latest read-event before the write-event
            latest_read_event = max(read_events, key=lambda x: x.timestamp)
            
            # Update pivot time for next previous task (second-latest write-event of current task)
            write_events.sort(key=lambda x: x.timestamp)
            if len(write_events) >= 2:
                current_pivot_event = write_events[-2]
                current_pivot_time = write_events[-2].timestamp  # Second-latest write event

            # Create task event pair
            execution_time = latest_write_event.timestamp - latest_read_event.timestamp
            task_pair = TaskEventPair(
                task_name=task_name,
                read_event=current_pivot_event,
                write_event=latest_write_event,
                execution_time=execution_time
            )
            task_sequence.insert(0, task_pair)  # Insert at beginning to maintain order
            
            
        
        # Create the estimated chain
        if len(task_sequence) > 1:  # Need at least 2 tasks for a meaningful chain
            total_chain_time = task_sequence[-1].write_event.timestamp - task_sequence[0].read_event.timestamp
            
            # Calculate chain length: z' event - z event (first task read-event)
            chain_length = z_prime_write_event.timestamp - current_pivot_event.timestamp
            
            # z is the first task's read-event
            z_event = current_pivot_event
            
            estimated_chain = CauseEffectChain(
                chain_id=f"estimated_chain_{len(task_sequence)}_tasks",
                task_sequence=task_sequence,
                total_chain_time=total_chain_time,
                chain_confidence=0.8,  # Estimated chains have lower confidence
                chain_type="estimated",
                chain_length=chain_length,
                z_event=z_event,  # Store z (first task's read-event) as the z_event
                z_prime_event=z_prime_write_event  # Store z' (last task's write-event) as the z_prime_event
            )
            
            # print(f"  Estimated chain created: {len(task_sequence)} tasks, total time: {total_chain_time}ms, chain length: {chain_length}ms")
            return [estimated_chain]
        else:
            print("  Insufficient tasks for estimated chain creation")
            return []
    


    def export_chain_comparison(self, result: EstimationResult, profile_name: str, run_id: int) -> None:
        """
        Export chain comparison results to CSV file
        
        Args:
            result: EstimationResult containing oracle and estimated chains
            profile_name: Name of the profile for the output filename
            run_id: Run ID for the output filename
        """
        import csv
        from pathlib import Path
        
        # Create output directory if it doesn't exist
        output_dir = Path("estimation_results")
        output_dir.mkdir(exist_ok=True)
        
        # Create CSV filename with profile name and run ID
        csv_filename = output_dir / f"chain_comparison_{profile_name}_run_{run_id:03d}.csv"
        
        try:
            with open(csv_filename, 'w', newline='') as csvfile:
                # Write header row
                header = ['Chain', 'Oracle', 'Estimation']
                writer = csv.writer(csvfile)
                writer.writerow(header)
                
                # Get the first oracle chain and estimated chain
                oracle_chain = result.oracle_chains[0] if result.oracle_chains else None
                estimated_chain = result.estimated_chains[0] if result.estimated_chains else None
                
                if oracle_chain and estimated_chain:
                    # Write z event row (second-earliest read-event of first task)
                    z_oracle = oracle_chain.z_event.timestamp if oracle_chain.z_event else ''
                    z_estimated = estimated_chain.z_event.timestamp if estimated_chain.z_event else ''
                    writer.writerow(['z', z_oracle, z_estimated])
                    
                    # Write task event pairs
                    max_tasks = max(len(oracle_chain.task_sequence), len(estimated_chain.task_sequence))
                    
                    for i in range(max_tasks):
                        # Oracle chain data
                        if i < len(oracle_chain.task_sequence):
                            oracle_pair = oracle_chain.task_sequence[i]
                            oracle_read = oracle_pair.read_event.timestamp if oracle_pair.read_event else ''
                            oracle_write = oracle_pair.write_event.timestamp if oracle_pair.write_event else ''
                        else:
                            oracle_read = ''
                            oracle_write = ''
                        
                        # Estimated chain data
                        if i < len(estimated_chain.task_sequence):
                            estimated_pair = estimated_chain.task_sequence[i]
                            estimated_read = estimated_pair.read_event.timestamp if estimated_pair.read_event else ''
                            estimated_write = estimated_pair.write_event.timestamp if estimated_pair.write_event else ''
                        else:
                            estimated_read = ''
                            estimated_write = ''
                        
                        # Write read-event row
                        task_name = f"task{i+1}" if i < len(oracle_chain.task_sequence) else f"task{i+1}"
                        writer.writerow([f"{task_name} read-event", oracle_read, estimated_read])
                        
                        # Write write-event row
                        writer.writerow([f"{task_name} write-event", oracle_write, estimated_write])
                
                else:
                    # If no chains available, write empty data
                    writer.writerow(['z', '', ''])
                    writer.writerow(['task1 read-event', '', ''])
                    writer.writerow(['task1 write-event', '', ''])
            
            # print(f"Chain comparison exported to: {csv_filename}")
            
        except Exception as e:
            print(f"Error exporting chain comparison: {e}")
    
    def print_estimation_summary(self, result: EstimationResult) -> None:
        """
        Print a summary of estimation results
        
        Args:
            result: EstimationResult to summarize
        """
        print(f"\n=== Cause-Effect Chain Estimation Summary ===")
        print(f"Simulation ID: {result.simulation_id}")
        print(f"Total tasks: {result.total_tasks}")
        print(f"Relationships found: {len(result.relationships)}")
        
        if result.relationships:
            print(f"\nTop relationships by confidence:")
            for i, rel in enumerate(result.relationships[:5]):
                print(f"  {i+1}. {rel.cause_task} -> {rel.effect_task} "
                      f"(confidence: {rel.confidence:.3f}, avg_delay: {rel.avg_delay:.2f}ms)")
        
        print(f"\nTask dependencies:")
        for task, deps in result.task_dependencies.items():
            if deps:
                print(f"  {task} -> {', '.join(deps)}")


def main():
    """Test function for the estimator"""
    print("Cause-Effect Chain Estimator")
    print("=" * 40)
    
    # Create estimator instance
    estimator = CauseEffectChainEstimator()
    
    # Test configuration
    print(f"Default confidence threshold: {estimator.config['min_confidence_threshold']}")
    print(f"Max delay threshold: {estimator.config['max_delay_threshold']}ms")
    print(f"Min evidence count: {estimator.config['min_evidence_count']}")
    
    print("\nEstimator initialized successfully!")
    print("Ready to analyze simulation results for cause-effect chains.")


if __name__ == "__main__":
    main()
