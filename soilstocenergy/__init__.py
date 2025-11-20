"""
SoilStocEnergy: Hierarchical Percolation Soil Water Model

A Python package for soil physics building on classical soil water physics
and extending with thermodynamic principles and dynamic connectivity.

This package implements a hierarchical thermodynamic percolation model that
explicitly accounts for dynamic connectivity changes in soil-water systems.
"""

__version__ = "0.1.0"
__author__ = "CAOS Research Group"

# Import main classes for easy access
from soilstocenergy.core.thermodynamics import FreeEnergyCalculator, LocalEquilibrium
from soilstocenergy.core.connectivity import ConnectivityCalculator, ConnectivityState

__all__ = [
    "FreeEnergyCalculator",
    "LocalEquilibrium",
    "ConnectivityCalculator",
    "ConnectivityState",
]
