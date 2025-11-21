# Hierarchical Percolation Soil Water Model - Rigorous Development Plan

## Project Goal
Develop a physically-stochastic alternative to Richards equation using percolation theory and thermodynamics, validated against analytical solutions, numerical benchmarks, and experimental data.

## Core Scientific Questions
1. Can percolation networks reproduce Richards equation solutions when fully connected?
2. Does dynamic connectivity κ(E) capture threshold behavior in infiltration/runoff?
3. Is the approach computationally efficient for hillslope-scale simulations?
4. Can it predict preferential flow and transport better than continuum approaches?

---

## PHASE 0: Baseline Richards Equation Solver (2 weeks)

### Objective
Implement robust 1D Richards equation solver as reference for all subsequent validation.

### Mathematical Formulation
**Mixed form Richards equation**:
```
∂θ/∂t = ∂/∂z[K(ψ)(∂ψ/∂z + 1)] - S(z,t)

where:
- θ(ψ): water content (Van Genuchten)
- K(ψ): hydraulic conductivity (Mualem-van Genuchten)
- S(z,t): sink term (ET, root uptake)
```

### Implementation Tasks

#### Task 0.1: Van Genuchten Model (Complete)
**File**: `soilstocenergy/baseline/retention.py`

```python
class VanGenuchtenComplete:
    """Complete Van Genuchten model with all functions."""

    def __init__(self, theta_r, theta_s, alpha, n, K_sat, l=0.5):
        self.theta_r = theta_r
        self.theta_s = theta_s
        self.alpha = alpha  # [1/m]
        self.n = n
        self.m = 1 - 1/n
        self.K_sat = K_sat  # [m/s]
        self.l = l  # Mualem pore connectivity parameter

    def water_content(self, psi):
        """Pressure head to water content (θ = θ(ψ))."""
        Se = (1 + abs(alpha * psi)**n)**(-m)
        return theta_r + (theta_s - theta_r) * Se

    def pressure_head(self, theta):
        """Water content to pressure head (ψ = ψ(θ)) - INVERSE."""
        Se = (theta - theta_r) / (theta_s - theta_r)
        Se = np.clip(Se, 1e-10, 1.0)
        psi = -1/alpha * (Se**(-1/m) - 1)**(1/n)
        return psi

    def specific_water_capacity(self, psi):
        """C(ψ) = dθ/dψ - needed for numerical solver."""
        if psi >= 0:
            return 0.0
        Se = (1 + abs(alpha * psi)**n)**(-m)
        dSe_dpsi = -m * n * alpha**n * abs(psi)**(n-1) * (1 + abs(alpha*psi)**n)**(-m-1)
        return (theta_s - theta_r) * dSe_dpsi

    def hydraulic_conductivity(self, psi):
        """K(ψ) - Mualem-van Genuchten model."""
        if psi >= 0:
            return K_sat
        Se = (1 + abs(alpha * psi)**n)**(-m)
        K = K_sat * Se**l * (1 - (1 - Se**(1/m))**m)**2
        return K

    def hydraulic_conductivity_theta(self, theta):
        """K(θ) - for comparison."""
        psi = self.pressure_head(theta)
        return self.hydraulic_conductivity(psi)
```

**Tests**:
- [ ] θ(ψ(θ)) = θ (inverse consistency)
- [ ] ψ(θ(ψ)) = ψ (forward consistency)
- [ ] C(ψ) > 0 for ψ < 0
- [ ] K(0) = K_sat
- [ ] K(-∞) → 0
- [ ] Compare with published VG curves

#### Task 0.2: Richards Equation Solver
**File**: `soilstocenergy/baseline/richards_1d.py`

```python
class RichardsEquation1D:
    """1D Richards equation solver using method of lines."""

    def __init__(self, z_nodes, soil_properties, initial_psi):
        """
        Parameters
        ----------
        z_nodes : array (n_nodes,)
            Vertical coordinates [m], positive downward
        soil_properties : VanGenuchtenComplete
            Soil hydraulic properties
        initial_psi : array (n_nodes,)
            Initial pressure head [m]
        """
        self.z = z_nodes
        self.n_nodes = len(z_nodes)
        self.dz = np.diff(z_nodes)
        self.soil = soil_properties
        self.psi = initial_psi.copy()
        self.theta = np.array([soil_properties.water_content(p) for p in initial_psi])
        self.time = 0.0

    def residual(self, t, psi):
        """
        Right-hand side for ODE solver.

        Returns: dθ/dt for each node
        """
        theta = np.array([self.soil.water_content(p) for p in psi])
        C = np.array([self.soil.specific_water_capacity(p) for p in psi])

        # Calculate fluxes at interfaces (n_nodes - 1 interfaces)
        flux = np.zeros(self.n_nodes - 1)
        for i in range(self.n_nodes - 1):
            # Average K at interface (geometric mean)
            K_i = self.soil.hydraulic_conductivity(psi[i])
            K_ip1 = self.soil.hydraulic_conductivity(psi[i+1])
            K_interface = np.sqrt(K_i * K_ip1)

            # Gradient
            dpsi_dz = (psi[i+1] - psi[i]) / self.dz[i]

            # Darcy flux (positive downward)
            flux[i] = -K_interface * (dpsi_dz + 1.0)

        # Apply boundary conditions
        flux_top = self.bc_top(t)
        flux_bottom = self.bc_bottom(t)

        # Build full flux array
        flux_full = np.zeros(self.n_nodes + 1)
        flux_full[0] = flux_top
        flux_full[1:-1] = flux
        flux_full[-1] = flux_bottom

        # Storage change: dθ/dt = -∂q/∂z
        dtheta_dt = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            if i == 0:
                dtheta_dt[i] = -(flux_full[i+1] - flux_full[i]) / (self.dz[i]/2)
            elif i == self.n_nodes - 1:
                dtheta_dt[i] = -(flux_full[i+1] - flux_full[i]) / (self.dz[i-1]/2)
            else:
                dtheta_dt[i] = -(flux_full[i+1] - flux_full[i]) / ((self.dz[i-1] + self.dz[i])/2)

        # Convert to dpsi/dt using chain rule: dpsi/dt = (1/C) * dθ/dt
        dpsi_dt = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            if C[i] > 1e-10:
                dpsi_dt[i] = dtheta_dt[i] / C[i]
            else:
                dpsi_dt[i] = 0.0

        return dpsi_dt

    def solve(self, t_span, bc_top_func, bc_bottom_func, method='BDF'):
        """
        Solve Richards equation over time span.

        Parameters
        ----------
        t_span : tuple (t0, tf)
            Time span [s]
        bc_top_func : callable(t) -> flux [m/s]
            Top boundary condition (positive = infiltration)
        bc_bottom_func : callable(t) -> flux [m/s]
            Bottom boundary condition
        method : str
            ODE solver method ('BDF' for stiff, 'RK45' for non-stiff)
        """
        from scipy.integrate import solve_ivp

        self.bc_top = bc_top_func
        self.bc_bottom = bc_bottom_func

        sol = solve_ivp(
            fun=self.residual,
            t_span=t_span,
            y0=self.psi,
            method=method,
            dense_output=True,
            rtol=1e-6,
            atol=1e-8
        )

        return sol

    def check_mass_balance(self, sol, t_eval):
        """Verify mass balance over simulation."""
        cumulative_top = 0.0
        cumulative_bottom = 0.0

        for i in range(len(t_eval) - 1):
            dt = t_eval[i+1] - t_eval[i]
            cumulative_top += self.bc_top(t_eval[i]) * dt
            cumulative_bottom += self.bc_bottom(t_eval[i]) * dt

        storage_initial = np.sum(self.theta * self.dz)

        psi_final = sol.y[:, -1]
        theta_final = np.array([self.soil.water_content(p) for p in psi_final])
        storage_final = np.sum(theta_final * self.dz)

        expected_storage = storage_initial + cumulative_top - cumulative_bottom
        error = abs(storage_final - expected_storage) / expected_storage

        return error
```

**Tests**:
- [ ] Constant infiltration → steady-state profile (analytical)
- [ ] Drainage from saturation → exponential decay
- [ ] Mass balance error < 0.1% for all scenarios
- [ ] Convergence test (halve Δz, error should decrease)
- [ ] Comparison with Hydrus-1D benchmark

#### Task 0.3: Analytical Solution Validation
**File**: `soilstocenergy/baseline/analytical.py`

Implement:
- Philip infiltration solution (short times)
- Green-Ampt solution (constant input)
- Steady infiltration profile
- Exponential drainage

**Validation Criteria**:
- [ ] Richards solver matches Philip solution within 2% for t < 1 hour
- [ ] Richards solver matches Green-Ampt within 5%
- [ ] Steady profile: |flux_in - flux_out| < 1e-8 m/s
- [ ] Drainage: match exponential time constant

### Deliverables
- [ ] `RichardsEquation1D` class passing all tests
- [ ] Documentation of numerical method
- [ ] Validation report comparing with 3+ analytical solutions
- [ ] Benchmark dataset for future comparisons

### Critical Integration Checkpoint
**Before proceeding to Phase 1**: Richards solver must match ALL analytical solutions within documented error bounds.

---

## PHASE 1: Complete Thermodynamic Framework (2 weeks)

### Objective
Implement rigorous free energy formulation and validate energy conservation.

### Mathematical Formulation

**Total free energy**:
```
E_total = ∫[0 to z] ρ_w * g * ψ_matric(θ) dz' + ρ_w * g * z + ρ_w * g * HAND

where:
- First term: matric potential energy
- Second term: gravitational potential energy
- Third term: topographic potential energy (HAND)
```

**Energy gradient drives flow**:
```
q = -K(θ) * (∂E/∂z) / (ρ_w * g) = -K(θ) * (∂ψ/∂z + 1)
```

### Implementation Tasks

#### Task 1.1: Free Energy Calculator
**File**: `soilstocenergy/core/thermodynamics_complete.py`

```python
class FreeEnergyCalculator:
    """Thermodynamically rigorous free energy calculations."""

    def __init__(self, soil_properties):
        self.soil = soil_properties
        self.RHO_W = 1000.0  # kg/m³
        self.G = 9.81  # m/s²

    def matric_potential_energy(self, theta):
        """
        Matric potential energy: ∫[θ_r to θ] ψ(θ') dθ'

        For Van Genuchten:
        E_matric = ρ_w * g * ∫ ψ(θ) dθ

        This requires numerical integration.
        """
        from scipy.integrate import quad

        if theta <= self.soil.theta_r:
            return 0.0

        def integrand(theta_prime):
            psi = self.soil.pressure_head(theta_prime)
            return psi

        integral, _ = quad(integrand, self.soil.theta_r, theta)
        return self.RHO_W * self.G * integral

    def total_free_energy(self, theta, z, HAND):
        """
        Total free energy at depth z with topographic position HAND.

        E_total = E_matric(θ) + ρ_w*g*z + ρ_w*g*HAND
        """
        E_matric = self.matric_potential_energy(theta)
        E_gravity = self.RHO_W * self.G * z
        E_topo = self.RHO_W * self.G * HAND

        return E_matric + E_gravity + E_topo

    def energy_gradient(self, theta_z, z, dz):
        """
        Calculate energy gradient between nodes.

        ∂E/∂z ≈ (E(z+dz) - E(z)) / dz
        """
        E_lower = self.total_free_energy(theta_z[1], z + dz, 0)
        E_upper = self.total_free_energy(theta_z[0], z, 0)

        return (E_lower - E_upper) / dz

    def flux_from_energy_gradient(self, theta, dE_dz):
        """
        Calculate flux from energy gradient.

        q = -K(θ) * (∂E/∂z) / (ρ_w * g)
        """
        K = self.soil.hydraulic_conductivity_theta(theta)
        q = -K * dE_dz / (self.RHO_W * self.G)
        return q
```

**Tests**:
- [ ] Energy increases with moisture
- [ ] Energy increases with height
- [ ] Energy gradient matches pressure gradient
- [ ] Flux from energy = flux from Darcy
- [ ] Energy conserved in closed system

#### Task 1.2: Energy-Gradient Equivalence Test
**File**: `tests/test_thermodynamics_validation.py`

Compare:
1. Darcy flux: `q = -K(θ) * (∂ψ/∂z + 1)`
2. Energy flux: `q = -K(θ) * (∂E/∂z) / (ρ_w*g)`

**Validation**: Should match within numerical precision (< 1e-10) for all moisture states.

### Deliverables
- [ ] Complete thermodynamic framework
- [ ] Demonstration that energy approach reproduces Darcy's law
- [ ] Energy conservation proof for Richards solver
- [ ] Documentation of thermodynamic foundation

### Critical Integration Checkpoint
**Before proceeding**: Energy-based flux calculations must exactly match pressure-based calculations.

---

## PHASE 2: Percolation Network Physics (3 weeks)

### Objective
Connect percolation theory to actual hydraulic conductivity and flow calculations.

### Mathematical Formulation

**Percolation-based effective conductivity**:
```
K_eff(p) = K_sat * p^μ * (p - p_c)^t   for p > p_c
         = 0                             for p ≤ p_c

where:
- p: fraction of active bonds
- p_c: percolation threshold (0.5927 for 2D square lattice)
- μ, t: critical exponents (universal)
```

**Bond activation probability**:
```
p_bond = P(E_free > E_crit) = κ(E_free, E_crit)
```

### Implementation Tasks

#### Task 2.1: Percolation Conductivity Model
**File**: `soilstocenergy/percolation/conductivity.py`

```python
class PercolationConductivity:
    """Effective conductivity from percolation theory."""

    def __init__(self, K_sat, p_c=0.5927, mu=1.3, t=1.0):
        """
        Parameters
        ----------
        K_sat : float
            Saturated conductivity [m/s]
        p_c : float
            Percolation threshold (0.5927 for 2D, 0.3116 for 3D)
        mu, t : float
            Critical exponents
        """
        self.K_sat = K_sat
        self.p_c = p_c
        self.mu = mu
        self.t = t

    def effective_conductivity(self, p):
        """
        Calculate K_eff from activation fraction p.

        Scaling law near percolation threshold.
        """
        if p <= self.p_c:
            return 0.0

        # Below saturation
        if p < 1.0:
            K_eff = self.K_sat * p**self.mu * (p - self.p_c)**self.t
        else:
            K_eff = self.K_sat

        return K_eff

    def conductivity_from_connectivity(self, kappa_field):
        """
        Calculate effective K from spatial connectivity field.

        Parameters
        ----------
        kappa_field : array
            Connectivity at each node (0 to 1)

        Returns
        -------
        K_eff : float
            Effective conductivity
        """
        # Mean-field approximation
        p_mean = np.mean(kappa_field)
        return self.effective_conductivity(p_mean)
```

**Tests**:
- [ ] K_eff = 0 for p < p_c
- [ ] K_eff = K_sat for p = 1
- [ ] Power-law behavior near p_c
- [ ] Compare with published percolation data

#### Task 2.2: Network Flow Solver
**File**: `soilstocenergy/percolation/network_flow.py`

```python
class NetworkFlowSolver:
    """Solve flow on percolation network."""

    def __init__(self, grid_shape, conductance_function):
        """
        Parameters
        ----------
        grid_shape : tuple
            (ny, nx) or (nz,) for network dimensions
        conductance_function : callable(theta, kappa) -> K
            Returns conductance for each bond
        """
        self.shape = grid_shape
        self.conductance_func = conductance_function
        self.network = self._create_network()

    def _create_network(self):
        """Create connectivity matrix."""
        # Implementation using networkx or sparse matrices
        pass

    def activate_bonds(self, kappa_field, threshold=0.5):
        """
        Activate bonds where κ > threshold.

        Returns active conductance matrix.
        """
        pass

    def solve_pressure_field(self, boundary_conditions):
        """
        Solve Laplace equation on active network.

        ∇·(K∇ψ) = 0 on active bonds

        Returns pressure at each node.
        """
        pass

    def calculate_fluxes(self, pressure_field):
        """Calculate flux through each bond using pressure solution."""
        pass
```

#### Task 2.3: Percolation-Richards Comparison
**File**: `tests/test_percolation_richards_comparison.py`

**Test Scenarios**:
1. **Fully connected** (κ = 1 everywhere):
   - Percolation network should give same result as Richards
   - Tolerance: < 1% difference in fluxes

2. **Random connectivity**:
   - Compare effective K with percolation scaling law
   - Verify power-law exponents

3. **Percolation threshold**:
   - Demonstrate sharp transition at p_c
   - Measure critical exponents

### Deliverables
- [ ] Percolation conductivity model validated
- [ ] Network flow solver reproducing Richards when fully connected
- [ ] Demonstration of critical behavior
- [ ] Quantitative comparison report

### Critical Integration Checkpoint
**Before proceeding**: Network solver must reproduce Richards solutions when all bonds active (within 1%).

---

## PHASE 3: Dynamic Connectivity Coupling (2 weeks)

### Objective
Implement feedback between moisture state and connectivity.

### Mathematical Formulation

**Connectivity function**:
```
κ(E_free, E_crit) = 1 / (1 + exp(-β(E_free - E_crit)))

where:
- β: sharpness of transition
- E_crit: critical energy threshold
```

**Coupled system**:
```
∂θ/∂t = -∂q/∂z    where q = -κ(E) · K(θ) · (∂ψ/∂z + 1)
κ = f(E(θ, z, HAND), E_crit)
```

### Implementation Tasks

#### Task 3.1: Connectivity Calculator (Proper Implementation)
**File**: `soilstocenergy/core/connectivity_complete.py`

```python
class ConnectivityCalculator:
    """Dynamic connectivity based on free energy."""

    def __init__(self, E_crit_base=-1000.0, beta=0.01, hysteresis=True):
        """
        Parameters
        ----------
        E_crit_base : float
            Base critical energy [J/m³]
        beta : float
            Transition sharpness [m³/J]
        hysteresis : bool
            Enable wetting-drying hysteresis
        """
        self.E_crit_base = E_crit_base
        self.beta = beta
        self.hysteresis = hysteresis

        if hysteresis:
            self.E_crit_wet = E_crit_base + 200.0  # Harder to activate when drying
            self.E_crit_dry = E_crit_base - 200.0  # Easier to stay active
            self.is_wetting = True

    def calculate_connectivity(self, E_free, mode='sigmoid'):
        """
        Calculate connectivity from free energy.

        Parameters
        ----------
        E_free : array
            Free energy at each node [J/m³]
        mode : str
            'sigmoid', 'step', or 'linear'

        Returns
        -------
        kappa : array
            Connectivity (0 to 1)
        """
        if not self.hysteresis:
            E_crit = self.E_crit_base
        else:
            E_crit = self.E_crit_wet if self.is_wetting else self.E_crit_dry

        if mode == 'sigmoid':
            kappa = 1.0 / (1.0 + np.exp(-self.beta * (E_free - E_crit)))
        elif mode == 'step':
            kappa = (E_free > E_crit).astype(float)
        elif mode == 'linear':
            kappa = np.clip((E_free - E_crit) / (2 * abs(E_crit)), 0, 1)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return kappa

    def update_hysteresis_state(self, theta_new, theta_old):
        """Update wetting/drying state based on moisture change."""
        if self.hysteresis:
            if np.mean(theta_new) > np.mean(theta_old):
                self.is_wetting = True
            else:
                self.is_wetting = False
```

**Tests**:
- [ ] κ = 0 for E << E_crit
- [ ] κ = 1 for E >> E_crit
- [ ] κ = 0.5 at E = E_crit (sigmoid)
- [ ] Hysteresis: E_crit_wet ≠ E_crit_dry
- [ ] β parameter controls transition sharpness

#### Task 3.2: Coupled θ-κ Solver
**File**: `soilstocenergy/coupled/coupled_solver.py`

```python
class CoupledSolver:
    """Solve coupled moisture-connectivity system."""

    def __init__(self, richards_solver, fe_calculator, conn_calculator):
        self.richards = richards_solver
        self.fe_calc = fe_calculator
        self.conn_calc = conn_calculator

    def residual_coupled(self, t, state):
        """
        Coupled residual function.

        state = [θ_1, ..., θ_n]

        Returns: dθ/dt considering dynamic κ
        """
        theta = state

        # Calculate free energy
        E_free = np.array([
            self.fe_calc.total_free_energy(theta[i], self.richards.z[i], 0)
            for i in range(len(theta))
        ])

        # Update connectivity
        kappa = self.conn_calc.calculate_connectivity(E_free)

        # Calculate fluxes with connectivity modifier
        # q = -κ * K(θ) * (∂ψ/∂z + 1)
        flux_modified = self.richards.calculate_flux(theta) * kappa

        # Storage change
        dtheta_dt = self._flux_divergence(flux_modified)

        return dtheta_dt

    def solve(self, t_span, boundary_conditions):
        """Solve coupled system."""
        from scipy.integrate import solve_ivp

        sol = solve_ivp(
            fun=self.residual_coupled,
            t_span=t_span,
            y0=self.richards.theta,
            method='BDF',  # Stiff solver
            rtol=1e-6,
            atol=1e-8
        )

        return sol
```

**Tests**:
- [ ] When κ = 1 everywhere: matches Richards solver
- [ ] When κ = 0 in some layers: no flux through those layers
- [ ] Mass balance maintained with variable κ
- [ ] Hysteresis affects wetting vs drying differently

### Deliverables
- [ ] Working coupled solver
- [ ] Demonstration of threshold behavior
- [ ] Comparison with Richards for limiting cases
- [ ] Hysteresis effect quantified

### Critical Integration Checkpoint
**Before proceeding**: Coupled solver must conserve mass and match Richards when κ = 1.

---

## PHASE 4: Topographic Integration (rDUNE) (2 weeks)

### Objective
Incorporate landscape position into critical energy thresholds.

### Mathematical Formulation

**rDUNE index**:
```
rDUNE = -ln(HAND / L_flow)

where:
- HAND: height above nearest drainage [m]
- L_flow: flow path length to drainage [m]
```

**Spatially variable E_crit**:
```
E_crit(x,y) = E_base - α_macro · macroporosity(x,y) - β_rDUNE · rDUNE(x,y)
```

### Implementation Tasks

#### Task 4.1: rDUNE Calculator
**File**: `soilstocenergy/topography/rdune.py`

```python
def calculate_rdune(HAND, flow_path_length, epsilon=0.01):
    """
    Calculate rDUNE index from topography.

    Parameters
    ----------
    HAND : array
        Height above nearest drainage [m]
    flow_path_length : array
        Flow path to drainage [m]
    epsilon : float
        Small value to avoid log(0)

    Returns
    -------
    rDUNE : array
        Energy dissipation index [-]
    """
    ratio = np.clip(HAND / flow_path_length, epsilon, 1.0)
    rDUNE = -np.log(ratio)
    return rDUNE

def calculate_spatial_E_crit(macroporosity, rDUNE, E_base, alpha_macro, beta_rDUNE):
    """
    Calculate spatially variable E_crit from soil structure and topography.

    Lower E_crit → easier to activate
    """
    E_crit = E_base - alpha_macro * macroporosity - beta_rDUNE * rDUNE
    return E_crit
```

**Tests**:
- [ ] rDUNE increases near stream (low HAND)
- [ ] E_crit lower near stream
- [ ] Macroporosity reduces E_crit
- [ ] Physical bounds maintained

#### Task 4.2: 2D Spatial Solver
**File**: `soilstocenergy/spatial/solver_2d.py`

Extend coupled solver to 2D with:
- Lateral fluxes between grid cells
- Spatially variable E_crit field
- Contributing area calculation

### Deliverables
- [ ] Spatial E_crit implementation
- [ ] 2D solver with lateral redistribution
- [ ] Demonstration of dynamic contributing area
- [ ] Validation against topographic wetness index

---

## PHASE 5: Comprehensive Validation (3 weeks)

### Validation Hierarchy

#### Level 1: Analytical Solutions
- [ ] Philip infiltration
- [ ] Green-Ampt
- [ ] Steady-state profiles
- [ ] Exponential drainage

**Acceptance**: < 5% error for all cases

#### Level 2: Numerical Benchmarks
- [ ] Hydrus-1D comparison (3 scenarios)
- [ ] Published Richards solutions
- [ ] Different soil types

**Acceptance**: < 10% error in fluxes and profiles

#### Level 3: Laboratory Data
- [ ] Infiltration experiments
- [ ] Column drainage
- [ ] Tracer breakthrough

**Acceptance**: Captures observed trends, R² > 0.8

#### Level 4: Field Data
- [ ] Hillslope runoff observations
- [ ] Saturated area dynamics
- [ ] Threshold responses

**Acceptance**: Reproduces threshold behavior

### Validation Report Structure
1. Test case description
2. Model setup
3. Results comparison
4. Error metrics
5. Physical interpretation
6. Computational cost

---

## PHASE 6: Particle Transport (2 weeks)

*Only after validated flow solver*

### Requirements
- Use solved velocity field: `v = q/θ`
- Only transport through connected regions (κ > threshold)
- Validate against advection-dispersion equation

---

## Critical Success Metrics

### Technical Metrics
- [ ] Mass balance error < 0.1% (all scenarios)
- [ ] Match Richards solutions within 5% when fully connected
- [ ] Demonstrate percolation threshold behavior
- [ ] Hysteresis effects quantified

### Scientific Metrics
- [ ] Reproduce analytical solutions
- [ ] Match experimental infiltration data
- [ ] Predict threshold runoff generation
- [ ] Capture preferential flow patterns

### Computational Metrics
- [ ] Faster than traditional Richards for same accuracy
- [ ] Scale to hillslope size (100x100 grid)
- [ ] Memory efficient

---

## Development Workflow

### For Each Phase:
1. **Design**: Mathematical formulation in markdown
2. **Implement**: Code with docstrings
3. **Test**: Unit tests for all functions
4. **Validate**: Compare with benchmarks
5. **Document**: Results and limitations
6. **Review**: Critical assessment before next phase

### Integration Checkpoints
- Phase 0 → 1: Richards solver validated
- Phase 1 → 2: Energy approach reproduces Darcy
- Phase 2 → 3: Network solver matches Richards
- Phase 3 → 4: Coupled solver conserves mass
- Phase 4 → 5: Spatial solver working
- Phase 5 → 6: All validation passed

### Version Control
- Branch per phase: `phase-N-description`
- Tag at checkpoints: `v0.N-checkpoint`
- Detailed commit messages with test results

---

## Timeline (16 weeks total)

| Phase | Duration | Key Deliverable |
|-------|----------|----------------|
| 0 | 2 weeks | Working Richards solver |
| 1 | 2 weeks | Thermodynamic framework |
| 2 | 3 weeks | Percolation network physics |
| 3 | 2 weeks | Coupled θ-κ solver |
| 4 | 2 weeks | Spatial/topographic integration |
| 5 | 3 weeks | Comprehensive validation |
| 6 | 2 weeks | Particle transport |

**Total**: 16 weeks of rigorous development

---

## Risk Mitigation

### Technical Risks
- **Richards solver instability**: Use proven BDF method, adaptive timesteps
- **Percolation network size**: Start with 1D, validate before scaling
- **Mass balance errors**: Check at every timestep, fail if > threshold
- **Computational cost**: Profile code, optimize bottlenecks

### Scientific Risks
- **Percolation may not capture physics**: Have Richards as fallback
- **Parameters hard to measure**: Sensitivity analysis, ensemble methods
- **Validation data limited**: Use multiple independent datasets

---

## Expected Outcomes

### If Successful
- Computationally efficient alternative to Richards
- Better representation of threshold behavior
- Natural emergence of preferential flow
- Published methodology and code

### If Partially Successful
- Hybrid approach (percolation + Richards)
- Identified limitations and improvements
- Framework for future development

### If Unsuccessful
- Documented why approach doesn't work
- Lessons for alternative formulations
- Still have working Richards solver
