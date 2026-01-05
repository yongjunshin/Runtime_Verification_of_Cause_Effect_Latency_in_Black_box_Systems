# Runtime Verification of Cause–Effect Latency in Black-Box Systems

This repository contains the experimental code and data used in the paper  
**“Runtime Verification of Cause–Effect Latency in Black-Box Systems”** (under review).

All experiments are synthetic and fully reproducible.

## Repository Structure

```text
Runtime_Verification_of_Cause_Effect_Latency_in_Black_box_Systems/
├── RQ1/        # Experiments for RQ1
├── RQ2-1/      # Impact of number of tasks (RQ2)
├── RQ2-2/      # Impact of task periods (RQ2)
├── RQ2-3/      # Impact of job utilization (RQ2)
├── RQ2-4/      # Visualization of job utilization impact (RQ2)
├── RQ3/        # Experiments for RQ3
├── RQ4-1/      # Estimation cost analysis (RQ4)
├── RQ4-2/      # Verification cost analysis (RQ4)
├── util/       # Simulator, estimator, verifier, and utility modules
└── venv/       # Virtual environment for experiments
```

## Usage

In each experiment directory:
- `experiment.py` runs the experiment and generates raw data.
- `analysis.py` performs analysis and produces figures.

Experiments can be executed independently for each research question.

## Notes

The codebase is under refinement to improve readability, including additional comments and minor cleanups.