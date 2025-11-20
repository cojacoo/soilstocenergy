# Hierarchical Percolation Soil Water Model - Implementation Plan

## Overview

This document outlines the detailed implementation plan for developing a hierarchical percolation soil water model that integrates thermodynamic principles, dynamic connectivity, and percolation theory.

## Design Philosophy

The model will integrate:
1. **Thermodynamic foundation** from the conceptual framework
2. **Particle-based tracking** inspired by echoRD
3. **Hierarchical layer structure** from LAST-model
4. **Dynamic connectivity** as the core innovation
5. **Modular Python architecture** building on soilwaterenergy

---

## Project Structure

```
soilstocenergy/
├── core/
│   ├── thermodynamics.py      # E_free, rDUNE, HAND calculations
│   ├── connectivity.py         # κ(E_free, E_crit) functions
│   ├── percolation.py         # Network topology, thresholds
│   └── water_balance.py       # Storage, fluxes for connected elements
├── models/
│   ├── vertical_1d.py         # Hierarchical soil column
│   ├── hillslope_2d.py        # Spatial percolation
│   └── particles.py           # Particle tracking (echoRD-inspired)
├── parameters/
│   ├── soil_properties.py     # Retention curves, K(θ)
│   ├── structure.py           # Macroporosity, E_crit relationships
│   └── topography.py          # DEM processing for rDUNE/HAND
├── utils/
│   ├── config.py              # JSON configuration (LAST-style)
│   ├── visualization.py       # Network plots, time series
│   └── io.py                  # Input/output handling
└── validation/
    ├── synthetic_cases.py     # Test scenarios
    └── metrics.py             # Performance evaluation
```

---

## Phase 1: Project Structure (Weeks 1-2)

### Objectives
- Set up modular Python package architecture
- Create core module structure
- Establish testing framework
- Set up configuration management

### Deliverables
- Package directory structure
- Basic `__init__.py` files
- `setup.py` or `pyproject.toml` for installation
- Testing framework with pytest
- Example configuration files

### Dependencies
- NumPy, SciPy (scientific computing)
- NetworkX or igraph (graph algorithms)
- Matplotlib (visualization)
- pytest (testing)
- JSON/YAML (configuration)

---

## Phase 2: Thermodynamic Core (Weeks 3-4)

### Components

#### 1. Free Energy Calculator
```python
E_free(x, t) = ψ_matric(θ(x,t)) + ρ_w * g * HAND(x)
```

**Features:**
- Input: water content θ, position x
- Output: Free energy [J/m³]
- Support for van Genuchten and Brooks-Corey retention curves
- Conversion utilities between θ, ψ, and E_free

#### 2. rDUNE Index
```python
rDUNE(x) = -ln(HAND(x) / flow_path_length(x))
```

**Features:**
- Computed from DEM preprocessing
- Stored as spatial field
- Controls topographic influence on E_crit
- Utility functions for DEM processing

#### 3. Local Equilibrium
**Features:**
- Define E_eq(HAND) for each position
- Storage excess: E_free > E_eq
- Storage deficit: E_free < E_eq
- Equilibrium state calculations

### Deliverables
- `thermodynamics.py` module with:
  - `FreeEnergyCalculator` class
  - `rDUNE` calculation functions
  - `LocalEquilibrium` class
  - Van Genuchten and Brooks-Corey implementations
- Unit tests for all energy calculations
- Example notebooks demonstrating concepts
- Documentation with equations and references

### Mathematical Foundations

**Van Genuchten Model:**
```
θ(ψ) = θ_r + (θ_s - θ_r) / [1 + (α|ψ|)^n]^m
ψ_m(θ) = -(1/α) * [(S_e^(-1/m) - 1)^(1/n)]
where S_e = (θ - θ_r)/(θ_s - θ_r)
```

**Free Energy:**
```
E_free = ψ_m(θ) + ρ_w * g * HAND
```

---

## Phase 3: Connectivity Framework (Weeks 5-6)

### Critical Energy Threshold

```python
E_crit(x) = f(macroporosity, rDUNE, structure_params)
```

### Implementation Options

#### 1. Simple Linear Model
```python
E_crit = E_base - α * macroporosity - β * rDUNE
```

#### 2. Sigmoid Transition (Gradual)
```python
κ(x,t) = 1 / (1 + exp(-β * (E_free(x,t) - E_crit(x))))
```

#### 3. Binary (Percolation Theory)
```python
κ(x,t) = Θ(E_free(x,t) - E_crit(x))  # Heaviside step function
```

### Parameter Relationships
- **Macropores** → lower E_crit (easier activation)
- **High rDUNE** → lower E_crit (favorable drainage position)
- **Structure damage** → higher E_crit (harder to connect)

### Connectivity State

**State Variable:** κ(x,t) ∈ [0, 1]
- κ = 0: Disconnected (no participation in active flow)
- κ = 1: Fully connected (active in network)
- 0 < κ < 1: Partially connected (gradual transition)

### Components

#### 1. ConnectivityCalculator Class
**Methods:**
- `calculate_E_crit(macroporosity, rDUNE, params)`: Calculate critical threshold
- `calculate_connectivity(E_free, E_crit, mode='binary')`: Calculate κ
- `update_connectivity(E_free_field)`: Update spatial connectivity field

#### 2. ConnectivityState Class
**Attributes:**
- `kappa`: Connectivity field κ(x,t)
- `E_crit`: Critical energy threshold field
- `is_active`: Boolean mask for active elements

**Methods:**
- `update(E_free)`: Update connectivity based on new energy state
- `get_active_fraction()`: Calculate fraction of connected domain
- `get_connectivity_pattern()`: Return spatial connectivity pattern

### Deliverables
- `connectivity.py` module with:
  - Multiple κ formulation options
  - `ConnectivityCalculator` class
  - `ConnectivityState` class
  - Calibration utilities for E_crit parameters
- Unit tests comparing binary vs. gradual transitions
- Sensitivity analysis tools for parameter effects
- Visualization functions for connectivity fields
- Documentation with theory and examples

### Key Features
- **Flexible formulations**: Easy to switch between binary and gradual
- **Parameter estimation**: Tools to infer E_crit from observations
- **Sensitivity analysis**: Explore effects of macroporosity, rDUNE
- **Visualization**: Spatial patterns of connectivity

---

## Phase 4: Percolation Network (Weeks 7-9)

### Network Representation
- **Nodes**: Soil elements with states (E_free, κ, θ)
- **Bonds**: Connections between elements
- **Bond activation**: `p_ij = κ_i * κ_j * Θ(E_interface > E_barrier)`

### Key Algorithms

#### 1. Cluster Identification (Union-Find)
- Find connected components
- Identify percolating clusters (spanning network)
- Track backbone vs. dead-ends

#### 2. Percolation Threshold Detection
```python
p_c = fraction_active_bonds_at_transition
```
- Monitor during wetting/drying
- Extract critical exponents

#### 3. Scaling Laws
```python
K_eff ∝ (p - p_c)^μ        # Conductivity
S_cluster ∝ (p - p_c)^(-γ)  # Cluster size
ξ ∝ (p - p_c)^(-ν)          # Correlation length
```

### Data Structures
- Sparse adjacency matrix for large networks
- NetworkX or igraph for analysis
- Efficient updates during state changes

### Deliverables
- `percolation.py` with network algorithms
- Visualization of active/inactive networks
- Critical exponent estimation tools

---

## Phase 5: 1D Vertical Model (Weeks 10-12)

### Soil Column Structure
```python
Column = [Layer_1, Layer_2, ..., Layer_N]
Layer_i:
  - depth range [z_top, z_bottom]
  - E_crit_i (structural property)
  - θ_i(t), E_free_i(t), κ_i(t)
  - macroporosity, matrix properties
```

### Water Balance for Connected Layers
```python
dE_free_i/dt = (Q_in - Q_out - ET_i - Drainage_i) * κ_i(t)
```

### Disconnected Layers
- κ ≈ 0 → no vertical drainage
- Only evaporation loss
- Storage in "trapped" zones

### Boundary Conditions
- **Top**: Precipitation input, infiltration excess if saturated
- **Bottom**: Free drainage or groundwater table

### Deliverables
- `vertical_1d.py` with hierarchical column class
- Example simulations: infiltration events, drainage
- Comparison with Richards equation solutions

---

## Phase 6: Hysteresis Mechanisms (Weeks 13-14)

### Wetting Path (Invasion Percolation)
- Elements activate when E_free > E_crit_wet
- Preferential filling of well-connected paths
- Cascade-like propagation

### Drying Path
- Elements deactivate when E_free < E_crit_dry
- E_crit_dry < E_crit_wet (hysteresis width)
- Network fragmentation
- Water trapped in disconnected clusters

### Trapping Rules
```python
if not connected_to_boundary(element_i):
    trapped_water_i = θ_i * volume_i
    drainage_i = 0
```

### Deliverables
- Hysteresis implementation in water balance
- Retention curve hysteresis comparison
- Trapped water quantification

---

## Phase 7: Particle Tracking Integration (Weeks 15-17)

### echoRD-Inspired Approach

#### 1. Particle Representation
```python
Particle:
  - position (layer or x,y,z)
  - age
  - tracer concentration
  - connected (True/False)
```

#### 2. Movement Rules
- Particles only move through **connected** network
- Advection along active bonds
- Dispersion within active clusters
- Immobilized in disconnected regions

#### 3. Flow Field Calculation
```python
v_i = K_eff(κ) * ∇E_free  # Only for κ > threshold
```

### Applications
- Tracer breakthrough curves
- Residence time distributions
- Preferential flow visualization

### Deliverables
- `particles.py` module
- Particle tracking in 1D column
- Breakthrough curve comparisons

---

## Phase 8: 2D/3D Hillslope Extension (Weeks 18-20)

### Spatial Network
- Grid cells as nodes
- rDUNE and HAND from DEM
- Lateral + vertical connectivity

### Connectivity Rules
```python
# Vertical (gravity-driven)
κ_vert(i→j) = κ(i) * κ(j) if j_below_i

# Lateral (pressure/saturation-driven)
κ_lat(i→j) = κ(i) * κ(j) * Θ(E_free_i > E_threshold_lat)
```

### Deliverables
- `hillslope_2d.py` with spatial percolation
- DEM preprocessing for rDUNE/HAND
- Animated visualizations of connectivity evolution

---

## Phase 9: Stochastic Ensemble Framework (Weeks 21-23)

### Monte Carlo Approach

#### 1. Parameter Sampling
- E_crit distributions (spatial uncertainty)
- Structural heterogeneity
- Initial conditions

#### 2. Ensemble Runs
```python
for realization in range(N_ensemble):
    E_crit_sample = sample_from_distribution()
    result = run_model(E_crit_sample)
    ensemble_results.append(result)
```

#### 3. Statistical Analysis
- PDF of soil moisture P(θ|rainfall)
- PDF of active area P(A_active|<E_free>)
- Uncertainty bounds

### Deliverables
- Ensemble simulation framework
- PDF generation tools
- Uncertainty quantification metrics

---

## Phase 10: Testing and Validation (Weeks 24-26)

### Synthetic Test Cases

1. **Homogeneous column:**
   - Should reproduce classical bucket model when E_crit → -∞
   - Verify mass balance

2. **Two-layer system:**
   - Test threshold behavior
   - Perched water table formation

3. **Macropore-matrix:**
   - Preferential vs. matrix flow
   - Compare with echoRD patterns

4. **Hillslope fill-and-spill:**
   - Threshold runoff generation
   - Active area expansion

### Validation Metrics
- Mass balance errors < 0.1%
- Percolation threshold within theoretical range
- Critical exponents match literature

### Deliverables
- Comprehensive test suite
- Benchmark results
- Documentation of model behavior

---

## Key Research Questions

### Theoretical
1. **What functional form for E_crit?** Linear, power-law, or empirical?
2. **Binary vs. gradual connectivity?** Which captures observations better?
3. **Hysteresis width?** How does E_crit_wet - E_crit_dry vary with soil type?

### Computational
1. **Efficiency?** Can we run ensembles in reasonable time?
2. **Spatial resolution?** What grid size needed to capture percolation?
3. **Time stepping?** How to handle rapid connectivity changes?

### Empirical
1. **Measurable E_crit?** Can we infer from tracer experiments?
2. **rDUNE validation?** Does it predict connectivity?
3. **Scale transfer?** Do universal laws hold across scales?

---

## Integration Strategy

The model will **build upon and connect**:
- **soilwaterenergy**: Provide thermodynamic calculation backend
- **echoRDmodel**: Adopt particle tracking for preferential flow
- **LAST-model**: Use layered structure and JSON config approach
- **New framework**: Add percolation theory and dynamic connectivity

**Modularity**: Each phase produces standalone components usable independently

**Extensibility**: Clear interfaces for:
- Alternative κ formulations
- Different soil property models
- Various boundary conditions
- Coupling with vegetation/energy balance

---

## Development Principles

1. **Test-Driven Development**: Write tests before implementation
2. **Documentation**: Docstrings, equations, and examples for all functions
3. **Version Control**: Regular commits with clear messages
4. **Code Review**: Validate correctness and clarity
5. **Reproducibility**: All results must be reproducible from code and config

---

## Success Criteria

### Phase 1-3
- [ ] Package structure established and installable
- [ ] Thermodynamic calculations validated against known solutions
- [ ] Connectivity framework tested with multiple formulations
- [ ] Clear documentation with examples

### Phase 4-6
- [ ] Percolation network algorithms working correctly
- [ ] 1D model reproduces expected threshold behavior
- [ ] Hysteresis captured in retention curves
- [ ] Mass balance maintained to high precision

### Phase 7-9
- [ ] Particle tracking integrated with connectivity
- [ ] 2D/3D spatial model operational
- [ ] Ensemble framework generates reasonable PDFs
- [ ] Scaling laws verified

### Phase 10
- [ ] All tests passing
- [ ] Comparison with classical models complete
- [ ] Performance acceptable for operational use
- [ ] Documentation comprehensive

---

## Timeline Summary

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| 1 | Weeks 1-2 | Project structure |
| 2 | Weeks 3-4 | Thermodynamic core |
| 3 | Weeks 5-6 | Connectivity framework |
| 4 | Weeks 7-9 | Percolation network |
| 5 | Weeks 10-12 | 1D vertical model |
| 6 | Weeks 13-14 | Hysteresis mechanisms |
| 7 | Weeks 15-17 | Particle tracking |
| 8 | Weeks 18-20 | 2D/3D hillslope |
| 9 | Weeks 21-23 | Stochastic ensemble |
| 10 | Weeks 24-26 | Testing & validation |

**Total Duration**: ~26 weeks (6 months)

---

## References

See `hierarchical_percolation_soil_water_model.md` for complete references and theoretical background.

---

*Document created: November 2024*
*Last updated: November 2024*
