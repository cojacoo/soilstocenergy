"""
Models module for soil water dynamics simulations.

This module provides different model implementations:
- 1D vertical soil column models
- 2D/3D hillslope models with spatial percolation
- Particle tracking for tracer and solute transport
"""

from soilstocenergy.models.vertical_1d import (
    SoilColumn1D,
    SoilLayer,
    create_uniform_column,
    create_layered_column,
)
from soilstocenergy.models.particles import (
    Particle,
    ParticleTracker,
)
from soilstocenergy.models.hillslope_2d import (
    Hillslope2D,
    HillslopeGrid,
    create_synthetic_hillslope,
)

__all__ = [
    "SoilColumn1D",
    "SoilLayer",
    "create_uniform_column",
    "create_layered_column",
    "Particle",
    "ParticleTracker",
    "Hillslope2D",
    "HillslopeGrid",
    "create_synthetic_hillslope",
]
