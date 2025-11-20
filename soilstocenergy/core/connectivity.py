"""
Connectivity framework for dynamic network activation.

This module implements the connectivity state tracking and critical energy
threshold calculations that form the core of the hierarchical percolation model.

Key concepts:
- Critical energy threshold E_crit(structure, rDUNE, position)
- Connectivity state κ(E_free, E_crit) ∈ [0, 1]
- Dynamic activation/deactivation of soil elements
"""

import numpy as np
from typing import Union, Callable, Optional, Dict, Any


class ConnectivityCalculator:
    """
    Calculator for connectivity state based on free energy and critical thresholds.

    The connectivity state κ ∈ [0, 1] determines whether a soil element
    participates in active hydrological processes.
    """

    def __init__(self,
                 transition_sharpness: float = 10.0,
                 default_mode: str = 'sigmoid'):
        """
        Initialize connectivity calculator.

        Parameters
        ----------
        transition_sharpness : float
            Sharpness parameter β for sigmoid transition [1/(J/m³)]
            Higher values = sharper transition
        default_mode : str
            Default connectivity mode: 'binary', 'sigmoid', or 'linear'
        """
        self.beta = transition_sharpness
        self.default_mode = default_mode

    def calculate_E_crit(self,
                        macroporosity: Union[float, np.ndarray] = 0.0,
                        rDUNE: Union[float, np.ndarray] = 0.0,
                        E_base: float = -1000.0,
                        alpha_macro: float = 500.0,
                        beta_rDUNE: float = 100.0) -> Union[float, np.ndarray]:
        """
        Calculate critical energy threshold for activation.

        E_crit = E_base - α_macro * macroporosity - β_rDUNE * rDUNE

        Parameters
        ----------
        macroporosity : float or array
            Macroporosity fraction [-]
            Higher macroporosity → lower E_crit (easier activation)
        rDUNE : float or array
            rDUNE topographic index [-]
            Higher rDUNE → lower E_crit (favorable drainage position)
        E_base : float
            Baseline critical energy [J/m³]
        alpha_macro : float
            Macroporosity effect coefficient [J/m³]
        beta_rDUNE : float
            rDUNE effect coefficient [J/m³]

        Returns
        -------
        float or array
            Critical energy threshold [J/m³]

        Notes
        -----
        - Macropores provide preferential flow paths → easier activation
        - High rDUNE indicates favorable topographic position → easier activation
        - Structural damage or compaction → higher E_crit
        """
        E_crit = E_base - alpha_macro * macroporosity - beta_rDUNE * rDUNE

        return E_crit

    def calculate_connectivity(self,
                              E_free: Union[float, np.ndarray],
                              E_crit: Union[float, np.ndarray],
                              mode: Optional[str] = None) -> Union[float, np.ndarray]:
        """
        Calculate connectivity state κ(E_free, E_crit).

        Parameters
        ----------
        E_free : float or array
            Free energy [J/m³]
        E_crit : float or array
            Critical energy threshold [J/m³]
        mode : str, optional
            Connectivity mode: 'binary', 'sigmoid', or 'linear'
            If None, uses default_mode

        Returns
        -------
        float or array
            Connectivity state κ ∈ [0, 1]

        Notes
        -----
        - κ = 0: Disconnected (no participation in active flow)
        - κ = 1: Fully connected (active in network)
        - 0 < κ < 1: Partial connection (gradual transitions)
        """
        mode = mode or self.default_mode

        if mode == 'binary':
            return self._binary_connectivity(E_free, E_crit)
        elif mode == 'sigmoid':
            return self._sigmoid_connectivity(E_free, E_crit)
        elif mode == 'linear':
            return self._linear_connectivity(E_free, E_crit)
        else:
            raise ValueError(f"Unknown connectivity mode: {mode}")

    def _binary_connectivity(self,
                           E_free: Union[float, np.ndarray],
                           E_crit: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Binary (Heaviside) connectivity function.

        κ = Θ(E_free - E_crit) = {1 if E_free > E_crit, 0 otherwise}

        This represents sharp percolation-theory activation.

        Parameters
        ----------
        E_free : float or array
            Free energy [J/m³]
        E_crit : float or array
            Critical energy threshold [J/m³]

        Returns
        -------
        float or array
            Connectivity state: 0 or 1
        """
        E_free = np.asarray(E_free)
        E_crit = np.asarray(E_crit)

        kappa = np.where(E_free > E_crit, 1.0, 0.0)

        return float(kappa) if kappa.ndim == 0 else kappa

    def _sigmoid_connectivity(self,
                            E_free: Union[float, np.ndarray],
                            E_crit: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Sigmoid (smooth) connectivity function.

        κ = 1 / (1 + exp(-β * (E_free - E_crit)))

        This provides a thermodynamically smooth transition.

        Parameters
        ----------
        E_free : float or array
            Free energy [J/m³]
        E_crit : float or array
            Critical energy threshold [J/m³]

        Returns
        -------
        float or array
            Connectivity state ∈ [0, 1]
        """
        E_free = np.asarray(E_free)
        E_crit = np.asarray(E_crit)

        # Clip to avoid numerical overflow in exp
        diff = np.clip(self.beta * (E_free - E_crit), -100, 100)
        kappa = 1.0 / (1.0 + np.exp(-diff))

        return float(kappa) if kappa.ndim == 0 else kappa

    def _linear_connectivity(self,
                           E_free: Union[float, np.ndarray],
                           E_crit: Union[float, np.ndarray],
                           width: float = 100.0) -> Union[float, np.ndarray]:
        """
        Linear transition connectivity function.

        κ = clip((E_free - E_crit + width/2) / width, 0, 1)

        Provides linear transition over a specified energy range.

        Parameters
        ----------
        E_free : float or array
            Free energy [J/m³]
        E_crit : float or array
            Critical energy threshold [J/m³]
        width : float
            Transition width [J/m³]

        Returns
        -------
        float or array
            Connectivity state ∈ [0, 1]
        """
        E_free = np.asarray(E_free)
        E_crit = np.asarray(E_crit)

        kappa = (E_free - E_crit + width/2) / width
        kappa = np.clip(kappa, 0.0, 1.0)

        return float(kappa) if kappa.ndim == 0 else kappa

    def calculate_bond_activation(self,
                                  kappa_i: Union[float, np.ndarray],
                                  kappa_j: Union[float, np.ndarray],
                                  E_barrier: float = 0.0,
                                  E_interface: Optional[Union[float, np.ndarray]] = None) -> Union[float, np.ndarray]:
        """
        Calculate bond activation probability between two elements.

        p_ij = κ_i * κ_j * Θ(E_interface - E_barrier)

        A bond is active when both connected elements are active and
        the interface energy exceeds the barrier.

        Parameters
        ----------
        kappa_i : float or array
            Connectivity of element i
        kappa_j : float or array
            Connectivity of element j
        E_barrier : float
            Energy barrier for bond activation [J/m³]
        E_interface : float or array, optional
            Interface energy [J/m³]
            If None, assumes E_interface > E_barrier

        Returns
        -------
        float or array
            Bond activation probability ∈ [0, 1]
        """
        p_bond = kappa_i * kappa_j

        if E_interface is not None:
            active = np.where(E_interface > E_barrier, 1.0, 0.0)
            p_bond = p_bond * active

        return p_bond


class ConnectivityState:
    """
    Track and manage spatial connectivity state over a domain.

    This class maintains the connectivity field κ(x,t) and provides
    methods for updating and analyzing the connectivity pattern.
    """

    def __init__(self,
                 shape: tuple,
                 calculator: Optional[ConnectivityCalculator] = None):
        """
        Initialize connectivity state.

        Parameters
        ----------
        shape : tuple
            Shape of spatial domain
        calculator : ConnectivityCalculator, optional
            Calculator for connectivity. If None, creates default.
        """
        self.shape = shape
        self.calculator = calculator or ConnectivityCalculator()

        # Initialize fields
        self.kappa = np.zeros(shape)  # Connectivity field
        self.E_crit = np.zeros(shape)  # Critical threshold field
        self.E_free = np.zeros(shape)  # Free energy field

    def set_E_crit(self,
                   E_crit: Union[float, np.ndarray],
                   macroporosity: Optional[np.ndarray] = None,
                   rDUNE: Optional[np.ndarray] = None,
                   **kwargs):
        """
        Set critical energy threshold field.

        Parameters
        ----------
        E_crit : float or array
            Critical energy threshold [J/m³]
            If float, uniform across domain
            If array, must match self.shape
        macroporosity : array, optional
            Macroporosity field for spatial variation
        rDUNE : array, optional
            rDUNE field for topographic control
        **kwargs
            Additional parameters for E_crit calculation
        """
        if macroporosity is not None and rDUNE is not None:
            # Calculate E_crit from structure and topography
            self.E_crit = self.calculator.calculate_E_crit(
                macroporosity=macroporosity,
                rDUNE=rDUNE,
                **kwargs
            )
        elif isinstance(E_crit, (int, float)):
            # Uniform E_crit
            self.E_crit = np.full(self.shape, E_crit)
        else:
            # Spatially variable E_crit provided
            E_crit = np.asarray(E_crit)
            if E_crit.shape != self.shape:
                raise ValueError(f"E_crit shape {E_crit.shape} does not match domain shape {self.shape}")
            self.E_crit = E_crit

    def update(self,
               E_free: np.ndarray,
               mode: Optional[str] = None):
        """
        Update connectivity state based on new free energy field.

        Parameters
        ----------
        E_free : array
            Free energy field [J/m³]
        mode : str, optional
            Connectivity mode to use
        """
        if E_free.shape != self.shape:
            raise ValueError(f"E_free shape {E_free.shape} does not match domain shape {self.shape}")

        self.E_free = E_free
        self.kappa = self.calculator.calculate_connectivity(
            E_free, self.E_crit, mode=mode
        )

    def get_active_fraction(self, threshold: float = 0.5) -> float:
        """
        Calculate fraction of domain that is active.

        Parameters
        ----------
        threshold : float
            Connectivity threshold for considering element "active"

        Returns
        -------
        float
            Fraction of active elements
        """
        return np.mean(self.kappa > threshold)

    def get_active_mask(self, threshold: float = 0.5) -> np.ndarray:
        """
        Get boolean mask of active elements.

        Parameters
        ----------
        threshold : float
            Connectivity threshold

        Returns
        -------
        array
            Boolean mask (True = active)
        """
        return self.kappa > threshold

    def get_connectivity_pattern(self) -> Dict[str, Any]:
        """
        Get summary statistics of connectivity pattern.

        Returns
        -------
        dict
            Dictionary with connectivity statistics:
            - 'mean': Mean connectivity
            - 'std': Standard deviation
            - 'active_fraction': Fraction with κ > 0.5
            - 'fully_active': Fraction with κ > 0.9
            - 'disconnected': Fraction with κ < 0.1
        """
        return {
            'mean': np.mean(self.kappa),
            'std': np.std(self.kappa),
            'active_fraction': self.get_active_fraction(threshold=0.5),
            'fully_active': self.get_active_fraction(threshold=0.9),
            'disconnected': np.mean(self.kappa < 0.1),
        }

    def identify_clusters(self, threshold: float = 0.5) -> tuple:
        """
        Identify connected clusters in the network.

        Parameters
        ----------
        threshold : float
            Connectivity threshold for considering elements connected

        Returns
        -------
        n_clusters : int
            Number of disconnected clusters
        cluster_labels : array
            Cluster labels for each element (-1 for inactive)

        Notes
        -----
        This is a placeholder. Full implementation requires proper
        cluster detection algorithm (e.g., connected components).
        """
        # This would require networkx or scipy for full implementation
        raise NotImplementedError("Cluster identification requires network analysis library")

    def get_percolation_probability(self) -> float:
        """
        Estimate probability of system-spanning percolation.

        Returns
        -------
        float
            Percolation probability [0, 1]

        Notes
        -----
        This is a simple approximation. Full implementation would
        check for actual spanning clusters.
        """
        # Simple approximation: if mean connectivity is high, likely percolating
        mean_kappa = np.mean(self.kappa)

        # Typical percolation threshold is ~0.59 for 3D, ~0.31 for 2D lattice
        if len(self.shape) == 3:
            p_c = 0.31  # 3D site percolation
        elif len(self.shape) == 2:
            p_c = 0.59  # 2D site percolation
        else:
            p_c = 0.5  # 1D

        # Rough estimate
        if mean_kappa > p_c:
            return min((mean_kappa - p_c) / (1 - p_c), 1.0)
        else:
            return max(mean_kappa / p_c * 0.1, 0.0)


class HysteresisConnectivity:
    """
    Connectivity calculator with hysteresis for wetting/drying cycles.

    Wetting and drying follow different paths:
    - Wetting: E_crit_wet (higher threshold)
    - Drying: E_crit_dry (lower threshold)
    """

    def __init__(self,
                 E_crit_wet: Union[float, np.ndarray],
                 E_crit_dry: Union[float, np.ndarray],
                 calculator: Optional[ConnectivityCalculator] = None):
        """
        Initialize hysteresis connectivity.

        Parameters
        ----------
        E_crit_wet : float or array
            Critical threshold for wetting [J/m³]
        E_crit_dry : float or array
            Critical threshold for drying [J/m³]
            Must have E_crit_dry < E_crit_wet
        calculator : ConnectivityCalculator, optional
            Base calculator
        """
        self.E_crit_wet = np.asarray(E_crit_wet)
        self.E_crit_dry = np.asarray(E_crit_dry)
        self.calculator = calculator or ConnectivityCalculator()

        # Validate hysteresis
        if np.any(self.E_crit_dry >= self.E_crit_wet):
            raise ValueError("E_crit_dry must be less than E_crit_wet for hysteresis")

        # Track current state
        self.is_wetting = True
        self.previous_E_free = None

    def calculate_connectivity(self,
                              E_free: Union[float, np.ndarray],
                              mode: Optional[str] = None) -> Union[float, np.ndarray]:
        """
        Calculate connectivity with hysteresis.

        Parameters
        ----------
        E_free : float or array
            Free energy [J/m³]
        mode : str, optional
            Connectivity mode

        Returns
        -------
        float or array
            Connectivity state κ
        """
        # Determine wetting or drying
        if self.previous_E_free is not None:
            self.is_wetting = E_free > self.previous_E_free

        # Select appropriate threshold
        E_crit = self.E_crit_wet if self.is_wetting else self.E_crit_dry

        # Calculate connectivity
        kappa = self.calculator.calculate_connectivity(E_free, E_crit, mode=mode)

        # Store state
        self.previous_E_free = E_free

        return kappa

    def get_hysteresis_width(self) -> Union[float, np.ndarray]:
        """
        Get hysteresis loop width.

        Returns
        -------
        float or array
            ΔE_crit = E_crit_wet - E_crit_dry [J/m³]
        """
        return self.E_crit_wet - self.E_crit_dry
