"""
Core module for thermodynamic and connectivity calculations.

This module provides the fundamental building blocks for the hierarchical
percolation soil water model, including:
- Free energy calculations
- rDUNE index computation
- Connectivity state tracking
- Percolation network analysis
"""

from soilstocenergy.core.thermodynamics import (
    FreeEnergyCalculator,
    LocalEquilibrium,
    VanGenuchten,
    BrooksCorey,
)
from soilstocenergy.core.connectivity import (
    ConnectivityCalculator,
    ConnectivityState,
    HysteresisConnectivity,
)
from soilstocenergy.core.percolation import (
    PercolationNetwork,
    UnionFind,
    ScalingLaws,
)

__all__ = [
    "FreeEnergyCalculator",
    "LocalEquilibrium",
    "VanGenuchten",
    "BrooksCorey",
    "ConnectivityCalculator",
    "ConnectivityState",
    "HysteresisConnectivity",
    "PercolationNetwork",
    "UnionFind",
    "ScalingLaws",
]
