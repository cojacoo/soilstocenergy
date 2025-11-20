"""
1D vertical soil column model with hierarchical percolation.

This module implements a layered soil column where each layer can be
active or inactive based on its connectivity state. Water balance is
solved only for connected layers.

Key features:
- Hierarchical layer structure
- Dynamic connectivity activation/deactivation
- Water balance with infiltration, drainage, evaporation
- Threshold behavior at critical energy states
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from soilstocenergy.core.thermodynamics import FreeEnergyCalculator
from soilstocenergy.core.connectivity import ConnectivityCalculator
from soilstocenergy.core.percolation import PercolationNetwork


@dataclass
class SoilLayer:
    """
    Properties of a single soil layer.

    Attributes
    ----------
    depth_top : float
        Top depth of layer [m]
    depth_bottom : float
        Bottom depth of layer [m]
    theta_r : float
        Residual water content [-]
    theta_s : float
        Saturated water content [-]
    alpha : float
        van Genuchten alpha [1/m]
    n : float
        van Genuchten n [-]
    K_sat : float
        Saturated hydraulic conductivity [m/s]
    macroporosity : float
        Macroporosity fraction [-]
    E_crit : float
        Critical energy threshold [J/m³]
    """
    depth_top: float
    depth_bottom: float
    theta_r: float
    theta_s: float
    alpha: float
    n: float
    K_sat: float
    macroporosity: float = 0.0
    E_crit: float = -1000.0

    @property
    def thickness(self) -> float:
        """Layer thickness [m]."""
        return self.depth_bottom - self.depth_top

    @property
    def depth_center(self) -> float:
        """Depth to layer center [m]."""
        return (self.depth_top + self.depth_bottom) / 2


class SoilColumn1D:
    """
    1D vertical soil column with dynamic connectivity.

    Water moves vertically through connected layers. Disconnected layers
    only lose water through evaporation.
    """

    def __init__(
        self,
        layers: List[SoilLayer],
        drainage_depth: float = 0.0,
        connectivity_mode: str = 'sigmoid'
    ):
        """
        Initialize soil column.

        Parameters
        ----------
        layers : list of SoilLayer
            Layer definitions (top to bottom)
        drainage_depth : float
            Depth to drainage (e.g., groundwater table) [m]
        connectivity_mode : str
            Connectivity mode: 'binary', 'sigmoid', or 'linear'
        """
        self.layers = layers
        self.n_layers = len(layers)
        self.drainage_depth = drainage_depth
        self.connectivity_mode = connectivity_mode

        # Initialize state variables
        self.theta = np.zeros(self.n_layers)  # Water content
        self.kappa = np.zeros(self.n_layers)  # Connectivity
        self.E_free = np.zeros(self.n_layers)  # Free energy
        self.fluxes = np.zeros(self.n_layers + 1)  # Fluxes at layer boundaries

        # Create calculators for each layer
        self.fe_calculators = []
        for layer in layers:
            calc = FreeEnergyCalculator(
                theta_r=layer.theta_r,
                theta_s=layer.theta_s,
                alpha=layer.alpha,
                n=layer.n,
                model='vanGenuchten'
            )
            self.fe_calculators.append(calc)

        # Connectivity calculator
        self.conn_calc = ConnectivityCalculator()

        # Percolation network (1D)
        self.network = PercolationNetwork(shape=(self.n_layers,))

        # Initialize with field capacity (assumed)
        self._initialize_to_field_capacity()

    def _initialize_to_field_capacity(self):
        """Initialize soil moisture to field capacity (approximation)."""
        for i, layer in enumerate(self.layers):
            # Field capacity ~ θ at ψ = -3.3 m (pF 2.5)
            psi_fc = -3.3
            self.theta[i] = self.fe_calculators[i].calculate_theta(psi_fc)

        self._update_energy_and_connectivity()

    def _update_energy_and_connectivity(self):
        """Update free energy and connectivity for all layers."""
        for i, layer in enumerate(self.layers):
            # Calculate HAND (height above drainage)
            HAND = self.drainage_depth - layer.depth_center

            # Free energy
            self.E_free[i] = self.fe_calculators[i].calculate_free_energy(
                self.theta[i], HAND
            )

            # Connectivity
            self.kappa[i] = self.conn_calc.calculate_connectivity(
                self.E_free[i],
                layer.E_crit,
                mode=self.connectivity_mode
            )

        # Update network state
        self.network.update_active_state(self.kappa, threshold=0.5)

    def calculate_conductivity(self, i: int) -> float:
        """
        Calculate hydraulic conductivity for layer i.

        Uses Mualem-van Genuchten model with connectivity modification:
        K = K_sat * Se^0.5 * [1 - (1 - Se^(1/m))^m]^2 * κ

        Parameters
        ----------
        i : int
            Layer index

        Returns
        -------
        float
            Hydraulic conductivity [m/s]
        """
        layer = self.layers[i]
        calc = self.fe_calculators[i]

        # Effective saturation
        Se = (self.theta[i] - layer.theta_r) / (layer.theta_s - layer.theta_r)
        Se = np.clip(Se, 0.0, 1.0)

        if Se < 1e-6:
            return 0.0

        # van Genuchten m parameter
        m = 1.0 - 1.0 / layer.n

        # Mualem-van Genuchten conductivity
        K_rel = Se**0.5 * (1.0 - (1.0 - Se**(1.0/m))**m)**2

        # Modified by connectivity
        K = layer.K_sat * K_rel * self.kappa[i]

        return K

    def calculate_fluxes(self):
        """
        Calculate water fluxes at layer boundaries.

        Uses Darcy's law with gravitational and pressure gradients.
        """
        self.fluxes[:] = 0.0

        for i in range(self.n_layers - 1):
            # Only calculate flux if both layers are connected
            if self.kappa[i] > 0.1 and self.kappa[i+1] > 0.1:
                # Average conductivity
                K_interface = 2.0 / (1.0 / (self.calculate_conductivity(i) + 1e-12) +
                                     1.0 / (self.calculate_conductivity(i+1) + 1e-12))

                # Hydraulic head gradient
                psi_i = self.fe_calculators[i].calculate_matric_potential(self.theta[i])
                psi_j = self.fe_calculators[i+1].calculate_matric_potential(self.theta[i+1])

                z_i = self.layers[i].depth_center
                z_j = self.layers[i+1].depth_center

                # Total head
                h_i = psi_i + z_i
                h_j = psi_j + z_j

                # Darcy flux (positive downward)
                dh_dz = (h_j - h_i) / (z_j - z_i)
                self.fluxes[i+1] = -K_interface * dh_dz

    def infiltrate(self, precip_rate: float, dt: float) -> float:
        """
        Apply precipitation and calculate infiltration.

        Parameters
        ----------
        precip_rate : float
            Precipitation rate [m/s]
        dt : float
            Time step [s]

        Returns
        -------
        float
            Actual infiltration [m] (may be less than precip if saturated)
        """
        # Potential infiltration
        infiltration_potential = precip_rate * dt

        # Top layer infiltration capacity
        top_layer = self.layers[0]
        storage_capacity = (top_layer.theta_s - self.theta[0]) * top_layer.thickness

        # Actual infiltration (limited by capacity)
        infiltration_actual = min(infiltration_potential, storage_capacity)

        # Update top layer only if connected
        if self.kappa[0] > 0.1:
            self.theta[0] += infiltration_actual / top_layer.thickness
        else:
            # If disconnected, water ponds or runs off
            infiltration_actual = 0.0

        # Excess becomes runoff
        runoff = infiltration_potential - infiltration_actual

        return infiltration_actual

    def evaporate(self, ET_rate: float, dt: float, root_depth: float = 0.5):
        """
        Apply evapotranspiration.

        Parameters
        ----------
        ET_rate : float
            Potential ET rate [m/s]
        dt : float
            Time step [s]
        root_depth : float
            Maximum root depth [m]
        """
        ET_potential = ET_rate * dt

        # Distribute ET over root zone
        for i, layer in enumerate(self.layers):
            if layer.depth_top < root_depth:
                # Fraction of root zone in this layer
                depth_in_roots = min(layer.depth_bottom, root_depth) - layer.depth_top
                fraction = depth_in_roots / root_depth

                # Available water
                available = (self.theta[i] - layer.theta_r) * layer.thickness

                # Actual ET from this layer
                ET_layer = min(ET_potential * fraction, available)

                # Update water content (works for both connected and disconnected)
                self.theta[i] -= ET_layer / layer.thickness

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
        # Update energy and connectivity
        self._update_energy_and_connectivity()

        # Infiltration
        if precip_rate > 0:
            self.infiltrate(precip_rate, dt)

        # Calculate fluxes
        self.calculate_fluxes()

        # Update water content for connected layers
        for i in range(self.n_layers):
            if self.kappa[i] > 0.1:  # Connected
                # Water balance: storage change = flux_in - flux_out
                flux_in = self.fluxes[i] if i > 0 else 0.0
                flux_out = self.fluxes[i+1] if i < self.n_layers - 1 else 0.0

                dtheta_dt = (flux_in - flux_out) / self.layers[i].thickness
                self.theta[i] += dtheta_dt * dt

                # Constrain to physical bounds
                self.theta[i] = np.clip(
                    self.theta[i],
                    self.layers[i].theta_r,
                    self.layers[i].theta_s
                )

        # Evapotranspiration
        if ET_rate > 0:
            self.evaporate(ET_rate, dt)

        # Final update
        self._update_energy_and_connectivity()

    def get_total_storage(self) -> float:
        """
        Calculate total water storage in column.

        Returns
        -------
        float
            Total storage [m]
        """
        storage = 0.0
        for i, layer in enumerate(self.layers):
            storage += self.theta[i] * layer.thickness
        return storage

    def get_profile(self) -> Dict[str, np.ndarray]:
        """
        Get soil profile data.

        Returns
        -------
        dict
            Dictionary with depth, theta, kappa, E_free arrays
        """
        depths = np.array([layer.depth_center for layer in self.layers])

        return {
            'depth': depths,
            'theta': self.theta.copy(),
            'kappa': self.kappa.copy(),
            'E_free': self.E_free.copy(),
            'fluxes': self.fluxes.copy(),
        }

    def is_percolating(self) -> bool:
        """
        Check if column has continuous connectivity from top to bottom.

        Returns
        -------
        bool
            True if percolating
        """
        return self.network.check_percolation()


def create_uniform_column(
    n_layers: int,
    total_depth: float,
    theta_r: float = 0.05,
    theta_s: float = 0.45,
    alpha: float = 2.0,
    n: float = 1.5,
    K_sat: float = 1e-5,
    macroporosity: float = 0.0,
    E_crit: float = -1000.0
) -> SoilColumn1D:
    """
    Create uniform soil column.

    Parameters
    ----------
    n_layers : int
        Number of layers
    total_depth : float
        Total column depth [m]
    theta_r : float
        Residual water content [-]
    theta_s : float
        Saturated water content [-]
    alpha : float
        van Genuchten alpha [1/m]
    n : float
        van Genuchten n [-]
    K_sat : float
        Saturated hydraulic conductivity [m/s]
    macroporosity : float
        Macroporosity fraction [-]
    E_crit : float
        Critical energy threshold [J/m³]

    Returns
    -------
    SoilColumn1D
        Initialized column
    """
    layer_thickness = total_depth / n_layers
    layers = []

    for i in range(n_layers):
        depth_top = i * layer_thickness
        depth_bottom = (i + 1) * layer_thickness

        layer = SoilLayer(
            depth_top=depth_top,
            depth_bottom=depth_bottom,
            theta_r=theta_r,
            theta_s=theta_s,
            alpha=alpha,
            n=n,
            K_sat=K_sat,
            macroporosity=macroporosity,
            E_crit=E_crit
        )
        layers.append(layer)

    return SoilColumn1D(layers)


def create_layered_column(
    layer_properties: List[Dict]
) -> SoilColumn1D:
    """
    Create layered soil column from property dictionaries.

    Parameters
    ----------
    layer_properties : list of dict
        List of dictionaries with layer properties

    Returns
    -------
    SoilColumn1D
        Initialized column
    """
    layers = []
    for props in layer_properties:
        layer = SoilLayer(**props)
        layers.append(layer)

    return SoilColumn1D(layers)
