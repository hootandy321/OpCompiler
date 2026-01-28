"""
Pytest configuration file for InfiniCore tests.

This file defines custom command line options and shared fixtures
that can be used across all test files in this directory.
"""

import pytest


def pytest_addoption(parser):
    """
    Add custom command line options for pytest.
    
    These options are available to all tests in this directory.
    Example usage:
        pytest bench_fusion.py --batch_size=64 --hidden_dim=2048
    """
    parser.addoption(
        "--batch_size", 
        action="store", 
        default=32, 
        type=int,
        help="Batch size for benchmark tests"
    )
    parser.addoption(
        "--hidden_dim", 
        action="store", 
        default=4096, 
        type=int,
        help="Hidden dimension for benchmark tests"
    )
    parser.addoption(
        "--warmup", 
        action="store", 
        default=50, 
        type=int,
        help="Number of warmup iterations for benchmark tests (increased for Triton autotuning)"
    )
    parser.addoption(
        "--runs", 
        action="store", 
        default=200, 
        type=int,
        help="Number of benchmark runs for stable measurements"
    )
