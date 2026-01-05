#!/usr/bin/env python3
"""
Random Profile Generator for Cause-Effect Chain Experiments

This module generates random YAML profiles for controlled experiments
with configurable period and utilization ranges.
"""

import yaml
import random
from pathlib import Path
from typing import Dict, Any, List


class RandomProfileGenerator:
    """Generates random cause-effect chain profiles for experiments"""
    
    def __init__(self, min_period: int, max_period: int, 
                 minimum_utilization: float, maximum_utilization: float):
        """
        Initialize the random profile generator
        
        Args:
            min_period: Minimum limit of randomly generated period (ms)
            max_period: Maximum limit of randomly generated period (ms)
            minimum_utilization: Minimum limit of WCET/period ratio (0 < x < 1)
            maximum_utilization: Maximum limit of WCET/period ratio (min_util < x < 1)
        """
        if min_period <= 1:
            raise ValueError("min_period must be > 1")
        if max_period < min_period:
            raise ValueError("max_period must be > min_period")
        if not (0 < minimum_utilization < 1):
            raise ValueError("minimum_utilization must be 0 < x < 1")
        if not (minimum_utilization <= maximum_utilization < 1):
            raise ValueError("maximum_utilization must be min_util < x < 1")
        
        self.min_period = min_period
        self.max_period = max_period
        self.minimum_utilization = minimum_utilization
        self.maximum_utilization = maximum_utilization
    
    def generate_profiles(self, task_count: int, num_profiles: int, 
                         duration: int, time_unit: str = "ms", seed: int = None, keyword: str = None, postfix: str = None) -> List[str]:
        """
        Generate random profiles and save them to files
        
        Args:
            task_count: Number of tasks in each chain profile
            num_profiles: Number of random profiles to generate
            duration: Fixed simulation duration
            time_unit: Time unit string (default: "ms")
            seed: Random seed for reproducible generation (optional)
            
        Returns:
            List of generated profile filenames (without path)
        """
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            print(f"Using random seed: {seed}")
        
        # Create output directory if it doesn't exist
        output_dir = Path("random_profiles")
        output_dir.mkdir(exist_ok=True)
        
        print(f"Generating {num_profiles} random profiles with {task_count} tasks each...")
        print(f"Period range: {self.min_period}-{self.max_period}ms")
        print(f"Utilization range: {self.minimum_utilization:.2f}-{self.maximum_utilization:.2f}")
        
        generated_profiles = []
        
        for profile_num in range(1, num_profiles + 1):
            if keyword == "Homo":
                profile_data = self._generate_single_homo_profile(task_count, duration, time_unit)
            else:
                profile_data = self._generate_single_profile(task_count, duration, time_unit)
            
            if keyword is not None:
                if postfix is not None:
                    filename = output_dir / f"{keyword}_random_profile_{task_count}tasks_{postfix}_{profile_num:03d}.yaml"
                else:
                    filename = output_dir / f"{keyword}_random_profile_{task_count}tasks_{profile_num:03d}.yaml"
            else:
                if postfix is not None:
                    filename = output_dir / f"random_profile_{task_count}tasks_{postfix}_{profile_num:03d}.yaml"
                else:
                    filename = output_dir / f"random_profile_{task_count}tasks_{profile_num:03d}.yaml"
            
            try:
                with open(filename, 'w') as file:
                    yaml.dump(profile_data, file, default_flow_style=False, sort_keys=False)
                print(f"  Generated: {filename}")
                generated_profiles.append(filename.name)  # Store just the filename
            except Exception as e:
                print(f"  Error saving {filename}: {e}")
        
        print(f"Profile generation completed. Files saved in: {output_dir}")
        return generated_profiles
    
    def _generate_single_profile(self, task_count: int, duration: int, time_unit: str) -> Dict[str, Any]:
        """
        Generate a single random profile
        
        Args:
            task_count: Number of tasks in the profile
            duration: Simulation duration
            time_unit: Time unit string
            
        Returns:
            Dictionary containing the profile data
        """
        profile = {
            'simulation': {
                'duration': duration,
                'time_unit': time_unit
            },
            'tasks': {
                'count': task_count
            }
        }
        
        # Generate random task profiles
        for i in range(1, task_count + 1):
            task_name = f"task{i}"
            
            # Random period within specified range
            if self.min_period == self.max_period:
                period = self.min_period
            else:   
                period = random.randint(self.min_period, self.max_period)
            
            # Random utilization within specified range
            if self.minimum_utilization == self.maximum_utilization:
                utilization = self.minimum_utilization
            else:
                utilization = random.uniform(self.minimum_utilization, self.maximum_utilization)
            
            # Calculate WCET based on period and utilization
            wcet = int(period * utilization)
            
            # Ensure WCET is at least 1 and doesn't exceed period
            wcet = max(1, min(wcet, period - 1))
            
            # Random phase (0 to period)
            phase = random.randint(0, period)
            
            profile['tasks'][task_name] = {
                'phase': phase,
                'period': period,
                'wcet': wcet
            }
        
        return profile
    
    def _generate_single_homo_profile(self, task_count: int, duration: int, time_unit: str) -> Dict[str, Any]:
        """
        Generate a single random profile
        
        Args:
            task_count: Number of tasks in the profile
            duration: Simulation duration
            time_unit: Time unit string
            
        Returns:
            Dictionary containing the profile data
        """
        profile = {
            'simulation': {
                'duration': duration,
                'time_unit': time_unit
            },
            'tasks': {
                'count': task_count
            }
        }
        # Generate a homo task period and utilization
        homo_period = random.randint(self.min_period, self.max_period)
        homo_utilization = random.uniform(self.minimum_utilization, self.maximum_utilization)
        homo_wcet = int(homo_period * homo_utilization)
        homo_wcet = max(1, min(homo_wcet, homo_period - 1))

        # Generate random task profiles
        for i in range(1, task_count + 1):
            task_name = f"task{i}"
            
            period = homo_period
            wcet = homo_wcet
            
            phase = random.randint(0, period)
            
            profile['tasks'][task_name] = {
                'phase': phase,
                'period': period,
                'wcet': wcet
            }
        
        return profile


def main():
    """Test function for the random profile generator"""
    print("Random Profile Generator")
    print("=" * 40)
    
    # Example usage
    generator = RandomProfileGenerator(
        min_period=50,
        max_period=500,
        minimum_utilization=0.3,
        maximum_utilization=0.8
    )
    
    # Generate some test profiles
    generator.generate_profiles(task_count=5, num_profiles=3, duration=2000)


if __name__ == "__main__":
    main()
