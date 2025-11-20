"""
Validation module for testing and benchmarking.

This module provides:
- Ensemble simulation framework
- Benchmark analytical solutions
- Validation metrics
- Model comparison tools
"""

from soilstocenergy.validation.ensemble import (
    ParameterDistribution,
    EnsembleSimulation,
    latin_hypercube_sampling,
)
from soilstocenergy.validation.benchmarks import (
    BenchmarkSuite,
    green_ampt_infiltration,
    philip_infiltration,
    percolation_theory_critical_exponents,
    validate_mass_balance,
)

__all__ = [
    "ParameterDistribution",
    "EnsembleSimulation",
    "latin_hypercube_sampling",
    "BenchmarkSuite",
    "green_ampt_infiltration",
    "philip_infiltration",
    "percolation_theory_critical_exponents",
    "validate_mass_balance",
]
