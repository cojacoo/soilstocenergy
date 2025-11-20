"""
Water balance calculations for connected soil elements.

This module implements water balance equations that respect connectivity:
- Storage changes only for connected elements
- Trapped water in disconnected regions
- Flux calculations with dynamic topology
"""

import numpy as np
from typing import Optional


def calculate_storage_change(
    theta_current: np.ndarray,
    flux_in: np.ndarray,
    flux_out: np.ndarray,
    ET: np.ndarray,
    thickness: np.ndarray,
    dt: float,
    kappa: np.ndarray,
    connectivity_threshold: float = 0.5
) -> np.ndarray:
    """
    Calculate storage change for connected elements.

    For connected elements (κ > threshold):
        dθ/dt = (flux_in - flux_out - ET) / thickness

    For disconnected elements (κ ≤ threshold):
        dθ/dt = -ET / thickness (only local losses)

    Parameters
    ----------
    theta_current : array
        Current water content [-]
    flux_in : array
        Influx at top boundary [m/s]
    flux_out : array
        Outflux at bottom boundary [m/s]
    ET : array
        Evapotranspiration rate [m/s]
    thickness : array
        Layer thickness [m]
    dt : float
        Time step [s]
    kappa : array
        Connectivity state [-]
    connectivity_threshold : float
        Threshold for considering element connected

    Returns
    -------
    array
        Updated water content [-]
    """
    theta_new = theta_current.copy()
    connected = kappa > connectivity_threshold

    # Connected elements: full water balance
    dtheta_connected = (flux_in - flux_out - ET) / thickness
    theta_new[connected] += dtheta_connected[connected] * dt

    # Disconnected elements: only ET loss
    dtheta_disconnected = -ET / thickness
    theta_new[~connected] += dtheta_disconnected[~connected] * dt

    return theta_new


def calculate_trapped_water(
    theta: np.ndarray,
    kappa: np.ndarray,
    thickness: np.ndarray,
    connectivity_threshold: float = 0.5
) -> float:
    """
    Calculate total trapped water in disconnected regions.

    Parameters
    ----------
    theta : array
        Water content [-]
    kappa : array
        Connectivity state [-]
    thickness : array
        Layer thickness [m]
    connectivity_threshold : float
        Threshold for considering element connected

    Returns
    -------
    float
        Total trapped water [m]
    """
    disconnected = kappa <= connectivity_threshold
    trapped = np.sum(theta[disconnected] * thickness[disconnected])

    return trapped


def calculate_mobile_water(
    theta: np.ndarray,
    kappa: np.ndarray,
    thickness: np.ndarray,
    connectivity_threshold: float = 0.5
) -> float:
    """
    Calculate total mobile (connected) water.

    Parameters
    ----------
    theta : array
        Water content [-]
    kappa : array
        Connectivity state [-]
    thickness : array
        Layer thickness [m]
    connectivity_threshold : float
        Threshold for considering element connected

    Returns
    -------
    float
        Total mobile water [m]
    """
    connected = kappa > connectivity_threshold
    mobile = np.sum(theta[connected] * thickness[connected])

    return mobile
