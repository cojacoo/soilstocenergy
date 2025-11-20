"""
2D hillslope model with spatial percolation.

This module implements a 2D soil-hillslope system where connectivity
varies spatially based on:
- Topography (HAND, rDUNE)
- Soil structure (macroporosity)
- Moisture state (free energy)

Key features:
- Spatial connectivity networks
- Fill-and-spill dynamics
- Lateral and vertical flow
- Dynamic contributing area
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

from soilstocenergy.core.thermodynamics import FreeEnergyCalculator
from soilstocenergy.core.connectivity import ConnectivityCalculator, ConnectivityState
from soilstocenergy.core.percolation import PercolationNetwork


@dataclass
class HillslopeGrid:
    """
    Grid properties for 2D hillslope.

    Attributes
    ----------
    ny : int
        Number of cells in y-direction (lateral)
    nx : int
        Number of cells in x-direction (distance from stream)
    dy : float
        Cell size in y [m]
    dx : float
        Cell size in x [m]
    elevation : array
        Surface elevation [m]
    HAND : array
        Height Above Nearest Drainage [m]
    flow_path_length : array
        Flow path length to nearest drainage [m]
    """
    ny: int
    nx: int
    dy: float
    dx: float
    elevation: np.ndarray
    HAND: np.ndarray
    flow_path_length: np.ndarray

    @property
    def shape(self) -> Tuple[int, int]:
        """Grid shape (ny, nx)."""
        return (self.ny, self.nx)

    @property
    def cell_area(self) -> float:
        """Cell area [m²]."""
        return self.dy * self.dx


class Hillslope2D:
    """
    2D hillslope model with spatial percolation.

    Represents a hillslope cross-section with spatial connectivity
    that evolves based on moisture state and topography.
    """

    def __init__(
        self,
        grid: HillslopeGrid,
        soil_properties: Dict,
        macroporosity: np.ndarray,
        E_crit_base: float = -1000.0,
        connectivity_mode: str = 'sigmoid'
    ):
        """
        Initialize 2D hillslope model.

        Parameters
        ----------
        grid : HillslopeGrid
            Grid structure and topography
        soil_properties : dict
            Soil hydraulic properties (theta_r, theta_s, alpha, n, K_sat)
        macroporosity : array
            Spatial macroporosity field [-]
        E_crit_base : float
            Base critical energy threshold [J/m³]
        connectivity_mode : str
            Connectivity mode
        """
        self.grid = grid
        self.soil_props = soil_properties
        self.macroporosity = macroporosity
        self.E_crit_base = E_crit_base
        self.connectivity_mode = connectivity_mode

        # State variables
        self.theta = np.full(grid.shape, soil_properties.get('theta_initial', 0.25))
        self.kappa = np.zeros(grid.shape)
        self.E_free = np.zeros(grid.shape)

        # Create calculators
        self.fe_calc = FreeEnergyCalculator(
            theta_r=soil_properties['theta_r'],
            theta_s=soil_properties['theta_s'],
            alpha=soil_properties['alpha'],
            n=soil_properties['n'],
            model='vanGenuchten'
        )

        self.conn_calc = ConnectivityCalculator()

        # Connectivity state manager
        self.conn_state = ConnectivityState(
            shape=grid.shape,
            calculator=self.conn_calc
        )

        # Percolation network
        self.network = PercolationNetwork(shape=grid.shape)

        # Calculate E_crit from topography and structure
        self._calculate_E_crit_field()

        # Initialize state
        self._update_energy_and_connectivity()

    def _calculate_E_crit_field(self):
        """Calculate spatial E_crit field from topography and structure."""
        from soilstocenergy.core.thermodynamics import calculate_rDUNE

        # Calculate rDUNE
        rDUNE = calculate_rDUNE(self.grid.HAND, self.grid.flow_path_length)

        # Set E_crit field
        self.conn_state.set_E_crit(
            E_crit=None,
            macroporosity=self.macroporosity,
            rDUNE=rDUNE,
            E_base=self.E_crit_base,
            alpha_macro=500.0,
            beta_rDUNE=100.0
        )

    def _update_energy_and_connectivity(self):
        """Update free energy and connectivity fields."""
        # Calculate free energy for all cells
        for i in range(self.grid.ny):
            for j in range(self.grid.nx):
                self.E_free[i, j] = self.fe_calc.calculate_free_energy(
                    self.theta[i, j],
                    self.grid.HAND[i, j]
                )

        # Update connectivity
        self.conn_state.update(self.E_free, mode=self.connectivity_mode)
        self.kappa = self.conn_state.kappa

        # Update network
        self.network.update_active_state(self.kappa, threshold=0.5)

    def calculate_conductivity(self) -> np.ndarray:
        """
        Calculate spatial hydraulic conductivity field.

        K = K_sat * K_rel(θ) * κ

        Returns
        -------
        array
            Hydraulic conductivity [m/s]
        """
        K_sat = self.soil_props['K_sat']

        # Effective saturation
        Se = (self.theta - self.soil_props['theta_r']) / \
             (self.soil_props['theta_s'] - self.soil_props['theta_r'])
        Se = np.clip(Se, 1e-6, 1.0)

        # Van Genuchten relative conductivity
        m = 1.0 - 1.0 / self.soil_props['n']
        K_rel = Se**0.5 * (1.0 - (1.0 - Se**(1.0/m))**m)**2

        # Modified by connectivity
        K = K_sat * K_rel * self.kappa

        return K

    def infiltrate(self, precip_rate: float, dt: float) -> np.ndarray:
        """
        Apply spatially uniform precipitation.

        Parameters
        ----------
        precip_rate : float
            Precipitation rate [m/s]
        dt : float
            Time step [s]

        Returns
        -------
        array
            Actual infiltration at each cell [m]
        """
        infiltration = np.zeros(self.grid.shape)
        precip_amount = precip_rate * dt

        for i in range(self.grid.ny):
            for j in range(self.grid.nx):
                # Only infiltrate where connected
                if self.kappa[i, j] > 0.5:
                    # Infiltration capacity
                    capacity = (self.soil_props['theta_s'] - self.theta[i, j]) * 0.1  # Assume 10cm depth
                    infiltration[i, j] = min(precip_amount, capacity)

                    # Update moisture
                    self.theta[i, j] += infiltration[i, j] / 0.1

        return infiltration

    def calculate_lateral_fluxes(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate lateral fluxes between cells.

        Uses Darcy's law with hydraulic head gradients.

        Returns
        -------
        flux_x : array
            Flux in x-direction [m³/s]
        flux_y : array
            Flux in y-direction [m³/s]
        """
        K = self.calculate_conductivity()

        # Hydraulic heads (matric + gravitational)
        psi = np.array([[self.fe_calc.calculate_matric_potential(self.theta[i, j])
                        for j in range(self.grid.nx)]
                       for i in range(self.grid.ny)])
        head = psi + self.grid.elevation

        # Fluxes in x-direction (distance from stream)
        flux_x = np.zeros((self.grid.ny, self.grid.nx - 1))
        for i in range(self.grid.ny):
            for j in range(self.grid.nx - 1):
                # Average conductivity at interface
                K_interface = 2.0 / (1.0/(K[i, j] + 1e-12) + 1.0/(K[i, j+1] + 1e-12))

                # Gradient
                dh_dx = (head[i, j+1] - head[i, j]) / self.grid.dx

                # Darcy flux (per unit width)
                flux_x[i, j] = -K_interface * dh_dx * self.grid.dy * 0.1  # [m³/s]

        # Fluxes in y-direction (lateral)
        flux_y = np.zeros((self.grid.ny - 1, self.grid.nx))
        for i in range(self.grid.ny - 1):
            for j in range(self.grid.nx):
                K_interface = 2.0 / (1.0/(K[i, j] + 1e-12) + 1.0/(K[i+1, j] + 1e-12))
                dh_dy = (head[i+1, j] - head[i, j]) / self.grid.dy
                flux_y[i, j] = -K_interface * dh_dy * self.grid.dx * 0.1  # [m³/s]

        return flux_x, flux_y

    def step(self, dt: float, precip_rate: float = 0.0, ET_rate: float = 0.0):
        """
        Advance model by one time step.

        Parameters
        ----------
        dt : float
            Time step [s]
        precip_rate : float
            Precipitation rate [m/s]
        ET_rate : float
            Evapotranspiration rate [m/s]
        """
        # Update state
        self._update_energy_and_connectivity()

        # Infiltration
        if precip_rate > 0:
            self.infiltrate(precip_rate, dt)

        # Lateral redistribution (simplified)
        flux_x, flux_y = self.calculate_lateral_fluxes()

        # Update theta based on fluxes (simplified water balance)
        # In full implementation, would solve continuity equation

        # ET loss
        if ET_rate > 0:
            ET_amount = ET_rate * dt
            for i in range(self.grid.ny):
                for j in range(self.grid.nx):
                    available = (self.theta[i, j] - self.soil_props['theta_r']) * 0.1
                    ET_actual = min(ET_amount, available)
                    self.theta[i, j] -= ET_actual / 0.1

        # Constrain theta
        self.theta = np.clip(
            self.theta,
            self.soil_props['theta_r'],
            self.soil_props['theta_s']
        )

        # Final update
        self._update_energy_and_connectivity()

    def get_contributing_area(self) -> float:
        """
        Calculate contributing area (connected to stream).

        Returns
        -------
        float
            Contributing area [m²]
        """
        # Cells connected to left boundary (stream)
        labels, _ = self.network.identify_clusters()

        # Check if percolating from right to left (hillslope to stream)
        # Simplified: count all active cells
        active_cells = np.sum(self.kappa > 0.5)

        return active_cells * self.grid.cell_area

    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Get current model state.

        Returns
        -------
        dict
            Dictionary with state arrays
        """
        return {
            'theta': self.theta.copy(),
            'kappa': self.kappa.copy(),
            'E_free': self.E_free.copy(),
            'HAND': self.grid.HAND,
            'elevation': self.grid.elevation,
        }


def create_synthetic_hillslope(
    ny: int = 30,
    nx: int = 50,
    dy: float = 2.0,
    dx: float = 2.0,
    slope: float = 0.1,
    soil_properties: Optional[Dict] = None
) -> Hillslope2D:
    """
    Create synthetic hillslope for testing.

    Parameters
    ----------
    ny : int
        Number of cells laterally
    nx : int
        Number of cells from stream
    dy : float
        Cell size lateral [m]
    dx : float
        Cell size longitudinal [m]
    slope : float
        Average hillslope gradient [-]
    soil_properties : dict, optional
        Soil properties

    Returns
    -------
    Hillslope2D
        Initialized hillslope model
    """
    # Create elevation field
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy

    X, Y = np.meshgrid(x, y)

    # Simple planar slope
    elevation = slope * X

    # HAND is distance from stream (left boundary)
    HAND = X * slope

    # Flow path length
    flow_path_length = X

    # Grid
    grid = HillslopeGrid(
        ny=ny,
        nx=nx,
        dy=dy,
        dx=dx,
        elevation=elevation,
        HAND=HAND,
        flow_path_length=flow_path_length
    )

    # Default soil properties
    if soil_properties is None:
        soil_properties = {
            'theta_r': 0.05,
            'theta_s': 0.45,
            'alpha': 2.0,
            'n': 1.5,
            'K_sat': 1e-5,
            'theta_initial': 0.25,
        }

    # Spatially variable macroporosity (higher near stream)
    macroporosity = 0.15 * np.exp(-X / 30.0)

    # Create hillslope
    hillslope = Hillslope2D(
        grid=grid,
        soil_properties=soil_properties,
        macroporosity=macroporosity,
        E_crit_base=-1500.0
    )

    return hillslope
