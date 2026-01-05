#!/usr/bin/env python3
"""
Cause-Effect Latency Verifier

This module implements the cause-effect latency verification algorithm using
sequential hypothesis testing with log-likelihood ratios.
"""

import numpy as np
from typing import List, Tuple, Literal, Optional
from dataclasses import dataclass
from scipy import stats
import math
from scipy.stats import norm, chi2


@dataclass
class VerificationResult:
    """Result of the cause-effect latency verification"""
    verdict: Literal['pass', 'fail', 'none']
    mean_latency: float
    num_used_samples: int
    num_passed_samples: int


def welford_update(n: int, x_bar: float, M2: float, x: float) -> Tuple[int, float, float]:
    """
    Welford's online algorithm for computing mean and variance
    
    Args:
        n: Current number of samples
        x_bar: Current mean
        M2: Current sum of squared differences
        x: New sample value
    
    Returns:
        Tuple of (updated_n, updated_mean, updated_M2)
    """
    n += 1
    delta = x - x_bar
    x_bar += delta / n
    delta2 = x - x_bar
    M2 += delta * delta2
    
    return n, x_bar, M2


# Cache for t-distribution quantiles to avoid repeated expensive calculations
_t_quantile_cache = {}

def clear_t_quantile_cache():
    """Clear the t-distribution quantile cache"""
    global _t_quantile_cache
    _t_quantile_cache.clear()

def howe_k_factor(u: float, confidence: float, nu: int, n: int) -> float:
    """
    Compute Howe's k-factor for normal tolerance intervals.
    
    Parameters
    ----------
    u : float
        Content proportion (coverage), e.g. 0.95 for 95%.
    gamma : float
        Confidence level, e.g. 0.99 for 99%.
    nu : int
        Degrees of freedom (usually n - 1).
    n : int
        Sample size.
        
    Returns
    -------
    k : float
        Howe's k factor.
    """
    # Input validation
    if not (0 < u < 1):
        raise ValueError("u (content) must be in (0,1)")
    if not (0 < confidence < 1):
        raise ValueError("gamma (confidence) must be in (0,1)")
    if nu <= 0:
        raise ValueError("nu (degrees of freedom) must be positive")
    if n <= 0:
        raise ValueError("n (sample size) must be positive")

    alpha = 1 - confidence
    
    # Use cache to avoid repeated expensive statistical calculations
    cache_key = (u, alpha, nu)
    if cache_key not in _t_quantile_cache:
        # Standard normal quantile for content u
        z_u = norm.ppf((1 + u) / 2.0)   # two-sided case
        
        # Chi-square critical value at gamma
        chi2_gamma = chi2.ppf(alpha, nu)
        
        # Store both values in cache
        _t_quantile_cache[cache_key] = (z_u, chi2_gamma)
    
    z_u, chi2_gamma = _t_quantile_cache[cache_key]
    
    # Howe's approximation
    k = math.sqrt((nu * (1 + 1/n) * z_u**2) / chi2_gamma)
    return k


def log_likelihood_ratio(sample: float, h0_mean: float, h1_mean: float, variance: float = 1.0) -> float:
    """
    Calculate the log-likelihood ratio for a single sample
    
    Args:
        sample: The observed sample value
        h0_mean: Mean under null hypothesis (H0)
        h1_mean: Mean under alternative hypothesis (H1)
        variance: Variance of the normal distribution (default: 1.0)
    
    Returns:
        Log-likelihood ratio value
    """
    # For normal distribution with known variance
    # Log-likelihood ratio = (1/(2*σ²)) * [(sample - μ₀)² - (sample - μ₁)²]
    # This simplifies to: (1/(2*σ²)) * (h1_mean - h0_mean) * (2*sample - h0_mean - h1_mean)
    
    return (1.0 / (2.0 * variance)) * (h1_mean - h0_mean) * (2.0 * sample - h0_mean - h1_mean)


def verify_cause_effect_latency(
    job_chain_length_samples: List[float],
    threshold: float,
    sensitivity_ratio: float,
    alpha: float,
    beta: float,
    variance: float = 1.0
) -> VerificationResult:
    """
    Verify cause-effect latency using sequential hypothesis testing
    
    Args:
        job_chain_length_samples: List of job chain length samples [l_hat_1, l_hat_2, ..., l_hat_k]
        threshold: Cause-effect latency requirement threshold (delta)
        sensitivity_ratio: Threshold sensitivity ratio (gamma) in (0,1)
        alpha: Type I error probability in (0,1)
        beta: Type II error probability in (0,1)
    
    Returns:
        VerificationResult containing verdict, mean latency, number of used samples, and number of passed samples
    """
    # Input validation
    if not (0 < sensitivity_ratio < 1):
        raise ValueError("sensitivity_ratio must be in (0,1)")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1)")
    if not (0 < beta < 1):
        raise ValueError("beta must be in (0,1)")
    if not job_chain_length_samples:
        raise ValueError("job_chain_length_samples cannot be empty")
    
    # Initialize parameters (Init section)
    h0 = threshold  # H0 mean
    h1 = (1 - sensitivity_ratio) * threshold  # H1 mean
    A = np.log((1 - beta) / alpha)  # Upper decision limit
    B = np.log(beta / (1 - alpha))  # Lower decision limit
    S = 0.0  # Cumulative log-likelihood ratio
    L_prime = []  # Set of used samples
    n = 0  # Number of passed samples
    psi = 'none'  # Verification verdict (no decision yet)
    
    # Process samples in reverse order (from k to 1)
    for i in range(len(job_chain_length_samples) - 1, -1, -1):
        l_hat = job_chain_length_samples[i]  # Get sample
        L_prime.append(l_hat)  # Append to used samples
        
        # Check if sample passes threshold
        if l_hat < threshold:
            n += 1  # Increment passed samples count
        
        # Update cumulative log-likelihood ratio
        S += log_likelihood_ratio(l_hat, h0, h1, variance)
        
        # Check decision criteria
        if S >= A:
            psi = 'pass'  # Latency < threshold
            break
        elif S <= B:
            psi = 'fail'  # Latency >= threshold
            break
    
    # Calculate mean of used samples
    mu = np.mean(L_prime) if L_prime else 0.0
    
    return VerificationResult(
        verdict=psi,
        mean_latency=mu,
        num_used_samples=len(L_prime),
        num_passed_samples=n
    )


def verify_cause_effect_latency_new(
    samples: List[float],
    threshold: float,
    content: float,
    confidence: float,
    min_samples: int = 3,
    max_samples: Optional[int] = None
) -> VerificationResult:
    """
    Verify cause-effect latency using tolerance intervals with Welford algorithm and Howe's k-factor
    
    Args:
        samples: List of latency samples [l_hat_1, l_hat_2, ..., l_hat_k]
        threshold: Upper limit threshold (delta)
        content: Content (coverage) parameter p in (0,1)
        confidence: Confidence level gamma in (0,1)
        min_samples: Minimum number of samples required (n_min >= 3)
        max_samples: Optional maximum number of samples (n_max >= n_min)
    
    Returns:
        VerificationResult containing verdict, mean latency, and sample counts
    """
    # Input validation
    if not (0 < content < 1):
        raise ValueError("content must be in (0,1)")
    if not (0 < confidence < 1):
        raise ValueError("confidence must be in (0,1)")
    if min_samples < 3:
        raise ValueError("min_samples must be >= 3")
    if max_samples is not None and max_samples < min_samples:
        raise ValueError("max_samples must be >= min_samples")
    if not samples:
        raise ValueError("samples cannot be empty")
    
    # Initialize parameters (Init section)
    n = 0
    x_bar = 0.0
    M2 = 0.0
    psi = 'none'  # inconclusive -> none
    latency_estimate = float('nan')
    
    # Process each sample
    for x in samples:
        # Update statistics using Welford's algorithm
        n, x_bar, M2 = welford_update(n, x_bar, M2, x)
        
        # Check if we have enough samples
        if n < min_samples:
            continue
        
        # Calculate tolerance interval parameters
        nu = n - 1  # degrees of freedom
        s = np.sqrt(M2 / nu) if nu > 0 else 0.0
        
        # Calculate Howe's k-factor for two-sided tolerance interval
        # Both upper and lower bounds use the same k-factor
        k = howe_k_factor(content, confidence, nu, n)
        
        # Calculate upper and lower tolerance limits
        U = x_bar + k * s
        L = x_bar - k * s
        # print (f"U: {U}, L: {L}")
        # Make decision based on tolerance limits
        
        if L <= threshold and U <= threshold:
            psi = 'pass'
            break
        elif L > threshold and U > threshold:
            psi = 'fail'
            break
        else:
            if n == max_samples:
                psi = 'fail'
                break



        
        
        # if U < threshold:
        #     psi = 'pass'  # safe -> pass (Latency <= threshold)
        #     # if n < max_samples:
        #     #     print(f"Early stop Pass: {U} < {threshold} at {n} samples {samples[:n]}")
        #     break
        # elif L > threshold:
        #     psi = 'fail'  # unsafe -> fail (Latency > threshold)
        #     # if n < max_samples:
        #     #     print(f"Early stop Fail: {L} > {threshold} at {n} samples {samples[:n]}")
        #     break
        # elif max_samples is not None and n >= max_samples:
        #     if U > threshold:
        #         psi = 'fail'
        #     break  # No decision (inconclusive -> none)
        # # else:
        # #     if n < max_samples:
        # #         print(f"Early stop None: {U} = {threshold} at {n} samples {samples[:n]}")
    
    # Calculate latency estimate if we have enough samples
    if n >= min_samples:
        nu = n - 1
        s = np.sqrt(M2 / nu) if nu > 0 else 0.0
        k = howe_k_factor(content, confidence, nu, n)
        latency_estimate = x_bar + k * s
    
    if psi == 'none':
        print("Error: Psi is none!")
    
    # print(psi)
    # print(n)
    return VerificationResult(
        verdict=psi,
        mean_latency=latency_estimate,
        num_used_samples=0,  # Not used in this algorithm
        num_passed_samples=0  # Not used in this algorithm
    )


def test_verification_algorithm():
    """
    Test function to demonstrate the verification algorithm
    """
    # Generate test data
    np.random.seed(42)
    
    # Test case 1: Samples mostly below threshold (should pass)
    samples_pass = np.random.normal(80, 10, 50).tolist()  # Mean=80, std=10, threshold=100
    
    # Test case 2: Samples mostly above threshold (should fail)
    samples_fail = np.random.normal(120, 10, 50).tolist()  # Mean=120, std=10, threshold=100
    
    # Test case 3: Mixed samples (uncertain)
    samples_mixed = np.random.normal(100, 15, 50).tolist()  # Mean=100, std=15, threshold=100
    
    # Parameters
    threshold = 100.0
    sensitivity_ratio = 0.1  # 10% sensitivity
    alpha = 0.05  # 5% Type I error
    beta = 0.1    # 10% Type II error
    
    print("=" * 80)
    print("CAUSE-EFFECT LATENCY VERIFICATION TEST")
    print("=" * 80)
    print(f"Threshold (δ): {threshold}")
    print(f"Sensitivity ratio (γ): {sensitivity_ratio}")
    print(f"Type I error (α): {alpha}")
    print(f"Type II error (β): {beta}")
    print()
    
    # Test case 1: Should pass
    print("Test Case 1: Samples mostly below threshold")
    print("-" * 50)
    result1 = verify_cause_effect_latency(samples_pass, threshold, sensitivity_ratio, alpha, beta)
    print(f"Verdict: {result1.verdict}")
    print(f"Mean latency: {result1.mean_latency:.2f}")
    print(f"Used samples: {result1.num_used_samples}")
    print(f"Passed samples: {result1.num_passed_samples}")
    print()
    
    # Test case 2: Should fail
    print("Test Case 2: Samples mostly above threshold")
    print("-" * 50)
    result2 = verify_cause_effect_latency(samples_fail, threshold, sensitivity_ratio, alpha, beta)
    print(f"Verdict: {result2.verdict}")
    print(f"Mean latency: {result2.mean_latency:.2f}")
    print(f"Used samples: {result2.num_used_samples}")
    print(f"Passed samples: {result2.num_passed_samples}")
    print()
    
    # Test case 3: Mixed (uncertain)
    print("Test Case 3: Mixed samples")
    print("-" * 50)
    result3 = verify_cause_effect_latency(samples_mixed, threshold, sensitivity_ratio, alpha, beta)
    print(f"Verdict: {result3.verdict}")
    print(f"Mean latency: {result3.mean_latency:.2f}")
    print(f"Used samples: {result3.num_used_samples}")
    print(f"Passed samples: {result3.num_passed_samples}")
    print()
    
    print("=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)


def test_verification_algorithm_new():
    """
    Test function to demonstrate the new verification algorithm using tolerance intervals
    """
    # Generate test data
    np.random.seed(42)
    
    # Test case 1: Samples mostly below threshold (should pass)
    samples_pass = np.random.normal(80, 10, 50).tolist()  # Mean=80, std=10, threshold=100
    
    # Test case 2: Samples mostly above threshold (should fail)
    samples_fail = np.random.normal(120, 10, 50).tolist()  # Mean=120, std=10, threshold=100
    
    # Test case 3: Mixed samples (uncertain)
    samples_mixed = np.random.normal(100, 15, 50).tolist()  # Mean=100, std=15, threshold=100
    
    # Parameters
    threshold = 100.0
    content = 0.95  # 95% content
    confidence = 0.95  # 95% confidence
    min_samples = 5
    max_samples = 30
    
    print("=" * 80)
    print("NEW CAUSE-EFFECT LATENCY VERIFICATION TEST")
    print("=" * 80)
    print(f"Threshold (δ): {threshold}")
    print(f"Content (p): {content}")
    print(f"Confidence (γ): {confidence}")
    print(f"Min samples: {min_samples}")
    print(f"Max samples: {max_samples}")
    print()
    
    # Test case 1: Should pass
    print("Test Case 1: Samples mostly below threshold")
    print("-" * 50)
    result1 = verify_cause_effect_latency_new(samples_pass, threshold, content, confidence, min_samples, max_samples)
    print(f"Verdict: {result1.verdict}")
    print(f"Latency estimate: {result1.mean_latency:.2f}")
    print(f"Used samples: {result1.num_used_samples}")
    print(f"Passed samples: {result1.num_passed_samples}")
    print()
    
    # Test case 2: Should fail
    print("Test Case 2: Samples mostly above threshold")
    print("-" * 50)
    result2 = verify_cause_effect_latency_new(samples_fail, threshold, content, confidence, min_samples, max_samples)
    print(f"Verdict: {result2.verdict}")
    print(f"Latency estimate: {result2.mean_latency:.2f}")
    print(f"Used samples: {result2.num_used_samples}")
    print(f"Passed samples: {result2.num_passed_samples}")
    print()
    
    # Test case 3: Mixed (uncertain)
    print("Test Case 3: Mixed samples")
    print("-" * 50)
    result3 = verify_cause_effect_latency_new(samples_mixed, threshold, content, confidence, min_samples, max_samples)
    print(f"Verdict: {result3.verdict}")
    print(f"Latency estimate: {result3.mean_latency:.2f}")
    print(f"Used samples: {result3.num_used_samples}")
    print(f"Passed samples: {result3.num_passed_samples}")
    print()
    
    # Test with different confidence levels
    print("Test Case 4: Different confidence levels")
    print("-" * 50)
    for conf in [0.8, 0.9, 0.95, 0.99, 0.999]:
        result = verify_cause_effect_latency_new(samples_pass, threshold, content, conf, min_samples, max_samples)
        print(f"Confidence {conf}: Verdict = {result.verdict}, Estimate = {result.mean_latency:.2f}")
    print()
    
    print("=" * 80)
    print("NEW TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    test_verification_algorithm()
    print("\n")
    test_verification_algorithm_new()
