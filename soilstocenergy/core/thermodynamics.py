"""
Thermodynamic calculations for soil water systems.

This module implements free energy calculations, retention curve models,
and thermodynamic equilibrium concepts for the hierarchical percolation
soil water model.

Key concepts:
- Free energy: E_free = ψ_matric + ρ_w * g * HAND
- Local equilibrium states
- rDUNE index for topographic control
"""

import numpy as np
from typing import Union, Tuple, Optional


class RetentionCurve:
    """
    Base class for soil water retention curves.

    Retention curves relate water content (θ) to matric potential (ψ).
    """

    def __init__(self, theta_r: float, theta_s: float):
        """
        Initialize retention curve parameters.

        Parameters
        ----------
        theta_r : float
            Residual water content [-]
        theta_s : float
            Saturated water content [-]
        """
        self.theta_r = theta_r
        self.theta_s = theta_s

    def theta_to_psi(self, theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert water content to matric potential."""
        raise NotImplementedError("Subclasses must implement theta_to_psi")

    def psi_to_theta(self, psi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert matric potential to water content."""
        raise NotImplementedError("Subclasses must implement psi_to_theta")

    def effective_saturation(self, theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate effective saturation.

        S_e = (θ - θ_r) / (θ_s - θ_r)

        Parameters
        ----------
        theta : float or array
            Volumetric water content [-]

        Returns
        -------
        float or array
            Effective saturation [-]
        """
        return (theta - self.theta_r) / (self.theta_s - self.theta_r)

    def theta_from_Se(self, Se: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate water content from effective saturation.

        θ = θ_r + S_e * (θ_s - θ_r)

        Parameters
        ----------
        Se : float or array
            Effective saturation [-]

        Returns
        -------
        float or array
            Volumetric water content [-]
        """
        return self.theta_r + Se * (self.theta_s - self.theta_r)


class VanGenuchten(RetentionCurve):
    """
    Van Genuchten (1980) retention curve model.

    θ(ψ) = θ_r + (θ_s - θ_r) / [1 + (α|ψ|)^n]^m

    where m = 1 - 1/n (Mualem condition)

    References
    ----------
    van Genuchten, M. Th. (1980). A closed-form equation for predicting the
    hydraulic conductivity of unsaturated soils. Soil Science Society of
    America Journal, 44(5), 892-898.
    """

    def __init__(self, theta_r: float, theta_s: float, alpha: float, n: float):
        """
        Initialize van Genuchten parameters.

        Parameters
        ----------
        theta_r : float
            Residual water content [-]
        theta_s : float
            Saturated water content [-]
        alpha : float
            van Genuchten parameter [1/m] or [1/cm]
        n : float
            van Genuchten parameter [-]
        """
        super().__init__(theta_r, theta_s)
        self.alpha = alpha
        self.n = n
        self.m = 1.0 - 1.0 / n  # Mualem condition

    def psi_to_theta(self, psi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert matric potential to water content (van Genuchten).

        Parameters
        ----------
        psi : float or array
            Matric potential [m] or [cm] (negative for unsaturated)

        Returns
        -------
        float or array
            Volumetric water content [-]
        """
        psi = np.asarray(psi)

        # For saturated conditions (ψ ≥ 0)
        theta = np.where(
            psi >= 0,
            self.theta_s,
            self.theta_r + (self.theta_s - self.theta_r) /
            np.power(1.0 + np.power(self.alpha * np.abs(psi), self.n), self.m)
        )

        return float(theta) if theta.ndim == 0 else theta

    def theta_to_psi(self, theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert water content to matric potential (van Genuchten).

        ψ = -(1/α) * [(S_e^(-1/m) - 1)^(1/n)]

        Parameters
        ----------
        theta : float or array
            Volumetric water content [-]

        Returns
        -------
        float or array
            Matric potential [m] or [cm] (negative for unsaturated)
        """
        theta = np.asarray(theta)

        # Effective saturation
        Se = self.effective_saturation(theta)
        Se = np.clip(Se, 1e-10, 1.0)  # Avoid numerical issues

        # For saturated conditions (θ ≥ θ_s)
        psi = np.where(
            theta >= self.theta_s,
            0.0,
            -(1.0 / self.alpha) * np.power(
                np.power(Se, -1.0/self.m) - 1.0,
                1.0/self.n
            )
        )

        return float(psi) if psi.ndim == 0 else psi


class BrooksCorey(RetentionCurve):
    """
    Brooks-Corey (1964) retention curve model.

    For |ψ| > ψ_b:  θ(ψ) = θ_r + (θ_s - θ_r) * (ψ_b / |ψ|)^λ
    For |ψ| ≤ ψ_b: θ(ψ) = θ_s

    References
    ----------
    Brooks, R. H., & Corey, A. T. (1964). Hydraulic properties of porous media.
    Hydrology Papers, Colorado State University.
    """

    def __init__(self, theta_r: float, theta_s: float, psi_b: float, lambda_: float):
        """
        Initialize Brooks-Corey parameters.

        Parameters
        ----------
        theta_r : float
            Residual water content [-]
        theta_s : float
            Saturated water content [-]
        psi_b : float
            Air entry pressure (bubbling pressure) [m] or [cm] (positive value)
        lambda_ : float
            Pore size distribution index [-]
        """
        super().__init__(theta_r, theta_s)
        self.psi_b = psi_b
        self.lambda_ = lambda_

    def psi_to_theta(self, psi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert matric potential to water content (Brooks-Corey).

        Parameters
        ----------
        psi : float or array
            Matric potential [m] or [cm] (negative for unsaturated)

        Returns
        -------
        float or array
            Volumetric water content [-]
        """
        psi = np.asarray(psi)
        psi_abs = np.abs(psi)

        # For |ψ| ≤ ψ_b: saturated
        # For |ψ| > ψ_b: unsaturated with power law
        theta = np.where(
            psi_abs <= self.psi_b,
            self.theta_s,
            self.theta_r + (self.theta_s - self.theta_r) *
            np.power(self.psi_b / psi_abs, self.lambda_)
        )

        return float(theta) if theta.ndim == 0 else theta

    def theta_to_psi(self, theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert water content to matric potential (Brooks-Corey).

        ψ = -ψ_b * S_e^(-1/λ)

        Parameters
        ----------
        theta : float or array
            Volumetric water content [-]

        Returns
        -------
        float or array
            Matric potential [m] or [cm]
        """
        theta = np.asarray(theta)

        # Effective saturation
        Se = self.effective_saturation(theta)
        Se = np.clip(Se, 1e-10, 1.0)

        # For saturated conditions
        psi = np.where(
            theta >= self.theta_s,
            0.0,
            -self.psi_b * np.power(Se, -1.0/self.lambda_)
        )

        return float(psi) if psi.ndim == 0 else psi


class FreeEnergyCalculator:
    """
    Calculator for soil water free energy.

    Free energy combines matric and gravitational potentials:
    E_free = ψ_matric + ρ_w * g * HAND

    where HAND is Height Above Nearest Drainage.
    """

    # Physical constants
    RHO_W = 1000.0  # Water density [kg/m³]
    G = 9.81        # Gravitational acceleration [m/s²]

    def __init__(self,
                 theta_r: float,
                 theta_s: float,
                 alpha: Optional[float] = None,
                 n: Optional[float] = None,
                 psi_b: Optional[float] = None,
                 lambda_: Optional[float] = None,
                 model: str = 'vanGenuchten'):
        """
        Initialize free energy calculator with retention curve model.

        Parameters
        ----------
        theta_r : float
            Residual water content [-]
        theta_s : float
            Saturated water content [-]
        alpha : float, optional
            van Genuchten alpha [1/m]
        n : float, optional
            van Genuchten n [-]
        psi_b : float, optional
            Brooks-Corey air entry pressure [m]
        lambda_ : float, optional
            Brooks-Corey pore size distribution index [-]
        model : str
            Retention curve model: 'vanGenuchten' or 'brooksCorey'
        """
        if model.lower() == 'vangenuchten':
            if alpha is None or n is None:
                raise ValueError("van Genuchten model requires 'alpha' and 'n' parameters")
            self.retention_curve = VanGenuchten(theta_r, theta_s, alpha, n)
        elif model.lower() == 'brookscorey':
            if psi_b is None or lambda_ is None:
                raise ValueError("Brooks-Corey model requires 'psi_b' and 'lambda_' parameters")
            self.retention_curve = BrooksCorey(theta_r, theta_s, psi_b, lambda_)
        else:
            raise ValueError(f"Unknown model: {model}. Use 'vanGenuchten' or 'brooksCorey'")

        self.model = model

    def calculate_free_energy(self,
                             theta: Union[float, np.ndarray],
                             HAND: Union[float, np.ndarray] = 0.0) -> Union[float, np.ndarray]:
        """
        Calculate free energy from water content and position.

        E_free = ψ_matric(θ) + ρ_w * g * HAND

        Parameters
        ----------
        theta : float or array
            Volumetric water content [-]
        HAND : float or array
            Height Above Nearest Drainage [m]

        Returns
        -------
        float or array
            Free energy [J/m³]
        """
        # Convert water content to matric potential [m]
        psi_matric = self.retention_curve.theta_to_psi(theta)

        # Gravitational potential [m]
        psi_gravity = HAND

        # Total free energy [J/m³] = [Pa] = [kg/(m·s²)]
        # ψ is in [m] of water head, so multiply by ρ_w * g to get [Pa]
        E_free = self.RHO_W * self.G * (psi_matric + psi_gravity)

        return E_free

    def calculate_matric_potential(self,
                                   theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate matric potential from water content.

        Parameters
        ----------
        theta : float or array
            Volumetric water content [-]

        Returns
        -------
        float or array
            Matric potential [m] (negative for unsaturated)
        """
        return self.retention_curve.theta_to_psi(theta)

    def calculate_theta(self,
                       psi_matric: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate water content from matric potential.

        Parameters
        ----------
        psi_matric : float or array
            Matric potential [m]

        Returns
        -------
        float or array
            Volumetric water content [-]
        """
        return self.retention_curve.psi_to_theta(psi_matric)

    def storage_state(self,
                     E_free: Union[float, np.ndarray],
                     E_equilibrium: Union[float, np.ndarray]) -> Union[str, np.ndarray]:
        """
        Determine storage state (excess or deficit) relative to equilibrium.

        Parameters
        ----------
        E_free : float or array
            Free energy [J/m³]
        E_equilibrium : float or array
            Equilibrium free energy [J/m³]

        Returns
        -------
        str or array
            'excess' if E_free > E_equilibrium (wet, draining)
            'deficit' if E_free < E_equilibrium (dry, under stress)
            'equilibrium' if E_free ≈ E_equilibrium
        """
        E_free = np.asarray(E_free)
        E_equilibrium = np.asarray(E_equilibrium)

        diff = E_free - E_equilibrium
        tol = 1.0  # Tolerance [J/m³]

        if E_free.ndim == 0:
            if diff > tol:
                return 'excess'
            elif diff < -tol:
                return 'deficit'
            else:
                return 'equilibrium'
        else:
            state = np.empty(E_free.shape, dtype=object)
            state[diff > tol] = 'excess'
            state[diff < -tol] = 'deficit'
            state[np.abs(diff) <= tol] = 'equilibrium'
            return state


class LocalEquilibrium:
    """
    Calculate local thermodynamic equilibrium states.

    At a given HAND position, there exists an equilibrium water content
    that depends on the local energy balance.
    """

    def __init__(self,
                 free_energy_calculator: FreeEnergyCalculator,
                 reference_theta: float = 0.30,
                 reference_HAND: float = 0.0):
        """
        Initialize local equilibrium calculator.

        Parameters
        ----------
        free_energy_calculator : FreeEnergyCalculator
            Calculator for free energy
        reference_theta : float
            Reference water content at reference position [-]
        reference_HAND : float
            Reference HAND (e.g., stream level) [m]
        """
        self.fe_calc = free_energy_calculator
        self.reference_theta = reference_theta
        self.reference_HAND = reference_HAND

        # Calculate reference free energy
        self.E_reference = self.fe_calc.calculate_free_energy(
            reference_theta, reference_HAND
        )

    def equilibrium_free_energy(self, HAND: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate equilibrium free energy at a given HAND.

        Assumes equilibrium is reached when the system adjusts to local conditions.

        Parameters
        ----------
        HAND : float or array
            Height Above Nearest Drainage [m]

        Returns
        -------
        float or array
            Equilibrium free energy [J/m³]
        """
        # Simple model: equilibrium shifts with gravitational potential
        delta_HAND = HAND - self.reference_HAND
        E_eq = self.E_reference + self.fe_calc.RHO_W * self.fe_calc.G * delta_HAND

        return E_eq

    def is_storage_excess(self,
                         theta: Union[float, np.ndarray],
                         HAND: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """
        Check if system is in storage excess regime.

        Parameters
        ----------
        theta : float or array
            Water content [-]
        HAND : float or array
            Height Above Nearest Drainage [m]

        Returns
        -------
        bool or array
            True if E_free > E_equilibrium
        """
        E_free = self.fe_calc.calculate_free_energy(theta, HAND)
        E_eq = self.equilibrium_free_energy(HAND)

        return E_free > E_eq

    def is_storage_deficit(self,
                          theta: Union[float, np.ndarray],
                          HAND: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """
        Check if system is in storage deficit regime.

        Parameters
        ----------
        theta : float or array
            Water content [-]
        HAND : float or array
            Height Above Nearest Drainage [m]

        Returns
        -------
        bool or array
            True if E_free < E_equilibrium
        """
        E_free = self.fe_calc.calculate_free_energy(theta, HAND)
        E_eq = self.equilibrium_free_energy(HAND)

        return E_free < E_eq


def calculate_rDUNE(HAND: Union[float, np.ndarray],
                    flow_path_length: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate rDUNE (reduced Dissipation per Unit length iNdEx).

    rDUNE = -ln(HAND / flow_path_length)

    The rDUNE index accounts for both the potential energy driver (HAND)
    and the dissipation along the flow path (flow_path_length).

    Parameters
    ----------
    HAND : float or array
        Height Above Nearest Drainage [m]
    flow_path_length : float or array
        Flow path length to nearest drainage [m]

    Returns
    -------
    float or array
        rDUNE index [-]

    References
    ----------
    Loritz, R., et al. (2019). A topographic index explaining hydrological
    similarity by accounting for the joint controls of runoff formation.
    Hydrology and Earth System Sciences, 23, 3807-3821.
    """
    HAND = np.asarray(HAND)
    flow_path_length = np.asarray(flow_path_length)

    # Avoid division by zero
    ratio = np.clip(HAND / flow_path_length, 1e-10, 1.0)

    rDUNE = -np.log(ratio)

    return float(rDUNE) if rDUNE.ndim == 0 else rDUNE


def calculate_HAND_from_elevation(elevation: np.ndarray,
                                  stream_elevation: Union[float, np.ndarray]) -> np.ndarray:
    """
    Calculate HAND from elevation data.

    HAND = elevation - stream_elevation

    Parameters
    ----------
    elevation : array
        Elevation [m]
    stream_elevation : float or array
        Nearest stream elevation [m]

    Returns
    -------
    array
        Height Above Nearest Drainage [m]
    """
    return elevation - stream_elevation
