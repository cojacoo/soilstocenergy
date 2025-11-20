# SoilStocEnergy: Hierarchical Percolation Soil Water Model

A Python package for soil physics building on classical soil water physics and extending with thermodynamic principles and dynamic connectivity.

## Overview

This package implements a hierarchical thermodynamic percolation model that explicitly accounts for dynamic connectivity changes in soil-water systems. Rather than assuming continuous hydraulic connectivity (as in the Richards equation), this framework recognizes that parts of the soil system can become functionally disconnected during drying.

## Key Concepts

- **Thermodynamic foundation**: Uses free energy (E_free = ψ_matric + ρgh) instead of water content
- **Dynamic connectivity**: Soil elements activate/deactivate based on energy thresholds
- **Percolation theory**: Network-based approach with threshold behavior
- **rDUNE index**: Topographic control on connectivity via reduced Dissipation per Unit length iNdEx
- **Hierarchical structure**: Self-similar across scales from pores to catchments

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from soilstocenergy import FreeEnergyCalculator, ConnectivityCalculator

# Create a free energy calculator with van Genuchten parameters
fe_calc = FreeEnergyCalculator(
    theta_r=0.05,  # Residual water content
    theta_s=0.45,  # Saturated water content
    alpha=2.0,     # van Genuchten alpha [1/m]
    n=1.5,         # van Genuchten n [-]
    model='vanGenuchten'
)

# Calculate free energy for a given water content and position
theta = 0.30  # Volumetric water content
HAND = 2.0    # Height Above Nearest Drainage [m]
E_free = fe_calc.calculate_free_energy(theta, HAND)

# Create connectivity calculator
conn_calc = ConnectivityCalculator()

# Calculate connectivity state
E_crit = -100.0  # Critical energy threshold [J/m³]
kappa = conn_calc.calculate_connectivity(E_free, E_crit, mode='sigmoid')

print(f"Free energy: {E_free:.2f} J/m³")
print(f"Connectivity: {kappa:.3f}")
```

## Documentation

For detailed documentation, see:
- [Implementation Plan](IMPLEMENTATION_PLAN.md) - Development roadmap and detailed design
- [Conceptual Framework](hierarchical_percolation_soil_water_model.md) - Theoretical foundation

## Project Structure

```
soilstocenergy/
├── core/              # Core thermodynamic and connectivity calculations
├── models/            # 1D, 2D, 3D model implementations
├── parameters/        # Soil properties and structure definitions
├── utils/             # Configuration, I/O, and visualization
├── validation/        # Test cases and benchmarks
└── tests/             # Unit and integration tests
```

## Development Status

This package is under active development. Current implementation status:

- [x] Phase 1: Project structure (Completed)
- [x] Phase 2: Thermodynamic core (Completed)
- [x] Phase 3: Connectivity framework (Completed)
- [x] Phase 4: Percolation network (Completed)
- [x] Phase 5: 1D vertical model (Completed)
- [ ] Phase 6: Hysteresis mechanisms
- [ ] Phase 7: Particle tracking
- [ ] Phase 8: 2D/3D hillslope extension
- [ ] Phase 9: Stochastic ensemble framework
- [ ] Phase 10: Testing and validation

## Related Work

This package builds on concepts from:
- [echoRDmodel](https://github.com/cojacoo/echoRDmodel) - Particle-based macropore-matrix interactions
- [LAST-model](https://github.com/KIT-HYD/last-model) - Layered soil system modeling
- CAOS Research Group - Thermodynamic hydrology framework

## References

Key publications:
- Zehe et al. (2019): Energy states of soil water - a thermodynamic perspective. HESS, 23, 971-987.
- Loritz et al. (2019): A topographic index explaining hydrological similarity (rDUNE). HESS, 23, 3807-3821.
- Hunt & Sahimi (2017): Percolation scaling, critical-path analysis. Reviews of Geophysics, 55, 993-1078.

See [hierarchical_percolation_soil_water_model.md](hierarchical_percolation_soil_water_model.md) for complete reference list.

## License

GPL-3.0 License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Contact

CAOS Research Group
Karlsruhe Institute of Technology (KIT)

## Acknowledgments

This work is part of the DFG's Catchments As Organized Systems (CAOS) research initiative.
