"""
Benchmark cases for model validation.

This module provides analytical solutions and test cases for validating
the hierarchical percolation model against known results.

Benchmarks include:
- Analytical infiltration solutions
- Bucket model comparison
- Richards equation comparison
- Percolation theory verification
"""

import numpy as np
from typing import Tuple, Optional


def green_ampt_infiltration(
    t: np.ndarray,
    K_sat: float,
    psi_f: float,
    theta_i: float,
    theta_s: float
) -> np.ndarray:
    """
    Green-Ampt infiltration solution.

    Analytical solution for infiltration into uniform soil.

    I(t) = K_sat * t + (theta_s - theta_i) * psi_f * ln(1 + I/(theta_s - theta_i)/psi_f)

    Parameters
    ----------
    t : array
        Time [s]
    K_sat : float
        Saturated hydraulic conductivity [m/s]
    psi_f : float
        Wetting front suction head [m] (positive)
    theta_i : float
        Initial water content [-]
    theta_s : float
        Saturated water content [-]

    Returns
    -------
    array
        Cumulative infiltration [m]
    """
    # Solve implicitly (simplified iterative solution)
    I = np.zeros_like(t)
    delta_theta = theta_s - theta_i

    for i, time in enumerate(t):
        if time <= 0:
            I[i] = 0
            continue

        # Iterative solution
        I_guess = K_sat * time
        for _ in range(10):  # 10 iterations
            I_new = K_sat * time + delta_theta * psi_f * np.log(1 + I_guess / (delta_theta * psi_f))
            if abs(I_new - I_guess) < 1e-6:
                break
            I_guess = I_new

        I[i] = I_guess

    return I


def philip_infiltration(
    t: np.ndarray,
    S: float,
    A: float
) -> np.ndarray:
    """
    Philip's two-term infiltration equation.

    I(t) = S * sqrt(t) + A * t

    Parameters
    ----------
    t : array
        Time [s]
    S : float
        Sorptivity [m/s^0.5]
    A : float
        Gravity term [m/s]

    Returns
    -------
    array
        Cumulative infiltration [m]
    """
    return S * np.sqrt(t) + A * t


def bucket_model_steady_state_pdf(
    s: np.ndarray,
    lambda_rain: float,
    alpha: float,
    n_: float,
    s_w: float,
    s_star: float,
    Z_r: float
) -> np.ndarray:
    """
    Rodriguez-Iturbe & Porporato steady-state soil moisture PDF.

    For exponential rainfall with intensity α and frequency λ.

    Parameters
    ----------
    s : array
        Relative saturation [-]
    lambda_rain : float
        Rainfall frequency [1/day]
    alpha : float
        Mean rainfall depth [mm]
    n_ : float
        Porosity * Z_r [mm]
    s_w : float
        Wilting point saturation [-]
    s_star : float
        Incipient stress point [-]
    Z_r : float
        Root depth [mm]

    Returns
    -------
    array
        Probability density [-]
    """
    # Simplified version (actual equation is more complex)
    # This is a placeholder for the concept

    gamma = n_ * Z_r / (lambda_rain * alpha)

    # Loss function parameters (simplified)
    eta_w = 0.01  # Evaporation rate

    # PDF (highly simplified)
    pdf = np.exp(-gamma * s / eta_w)
    pdf = pdf / np.trapz(pdf, s)  # Normalize

    return pdf


def theis_solution(
    r: np.ndarray,
    t: float,
    Q: float,
    T: float,
    S: float
) -> np.ndarray:
    """
    Theis solution for groundwater drawdown.

    Used for comparison with deep drainage.

    Parameters
    ----------
    r : array
        Radial distance from well [m]
    t : float
        Time since pumping started [s]
    Q : float
        Pumping rate [m³/s]
    T : float
        Transmissivity [m²/s]
    S : float
        Storage coefficient [-]

    Returns
    -------
    array
        Drawdown [m]
    """
    from scipy.special import exp1  # Exponential integral

    u = r**2 * S / (4 * T * t)
    drawdown = Q / (4 * np.pi * T) * exp1(u)

    return drawdown


def percolation_theory_critical_exponents() -> dict:
    """
    Theoretical critical exponents for percolation.

    Returns
    -------
    dict
        Critical exponents for 2D and 3D
    """
    return {
        '2d': {
            'nu': 4/3,       # Correlation length
            'beta': 5/36,    # Percolation strength
            'gamma': 43/18,  # Susceptibility/cluster size
            'mu': 1.3,       # Conductivity (approximate)
            'p_c': 0.5927,   # Site percolation threshold (square lattice)
        },
        '3d': {
            'nu': 0.88,
            'beta': 0.41,
            'gamma': 1.8,
            'mu': 2.0,
            'p_c': 0.3116,   # Site percolation threshold (cubic lattice)
        }
    }


def compare_with_bucket_model(
    percolation_result: dict,
    bucket_result: dict,
    metric: str = 'mean_storage'
) -> float:
    """
    Compare percolation model with bucket model.

    Parameters
    ----------
    percolation_result : dict
        Results from percolation model
    bucket_result : dict
        Results from bucket model
    metric : str
        Metric to compare

    Returns
    -------
    float
        Relative difference
    """
    perc_value = percolation_result.get(metric, 0)
    bucket_value = bucket_result.get(metric, 0)

    if bucket_value == 0:
        return np.nan

    relative_diff = abs(perc_value - bucket_value) / bucket_value

    return relative_diff


def validate_mass_balance(
    initial_storage: float,
    final_storage: float,
    total_input: float,
    total_output: float,
    tolerance: float = 0.01
) -> Tuple[bool, float]:
    """
    Validate mass balance closure.

    final = initial + input - output

    Parameters
    ----------
    initial_storage : float
        Initial water storage [m]
    final_storage : float
        Final water storage [m]
    total_input : float
        Total input (precipitation, etc.) [m]
    total_output : float
        Total output (ET, drainage, etc.) [m]
    tolerance : float
        Acceptable relative error

    Returns
    -------
    passes : bool
        Whether mass balance closes within tolerance
    error : float
        Relative error
    """
    expected_final = initial_storage + total_input - total_output
    error = abs(final_storage - expected_final) / max(abs(expected_final), 1e-10)

    passes = error < tolerance

    return passes, error


def analytical_steady_state_profile(
    z: np.ndarray,
    infiltration_rate: float,
    K_sat: float,
    theta_r: float,
    theta_s: float,
    alpha: float,
    n: float
) -> np.ndarray:
    """
    Analytical steady-state soil moisture profile.

    Under constant infiltration, using van Genuchten parameters.

    This is approximate - exact solution requires solving ODE.

    Parameters
    ----------
    z : array
        Depth [m]
    infiltration_rate : float
        Steady infiltration [m/s]
    K_sat : float
        Saturated conductivity [m/s]
    theta_r : float
        Residual water content [-]
    theta_s : float
        Saturated water content [-]
    alpha : float
        van Genuchten alpha [1/m]
    n : float
        van Genuchten n [-]

    Returns
    -------
    array
        Water content profile [-]
    """
    # Simplified: assume unit gradient (gravity drainage)
    # K(θ) = infiltration_rate

    # For steady state: K(θ) = q = constant
    # Need to invert K(θ) to get θ(z)

    # Approximate as exponential approach to equilibrium
    m = 1 - 1/n
    q_ratio = infiltration_rate / K_sat

    # Effective saturation needed to conduct q
    if q_ratio >= 1:
        Se = 1.0
    else:
        # Approximate inversion of Mualem-van Genuchten
        Se = q_ratio**(2.0/3.0)  # Simplified

    theta = theta_r + Se * (theta_s - theta_r)

    # Profile (could add depth dependence)
    profile = np.full_like(z, theta)

    return profile


class BenchmarkSuite:
    """
    Collection of benchmark tests for validation.
    """

    def __init__(self):
        """Initialize benchmark suite."""
        self.results = {}

    def run_infiltration_benchmark(
        self,
        model_function: callable,
        analytical_function: callable,
        **kwargs
    ) -> dict:
        """
        Compare model infiltration with analytical solution.

        Parameters
        ----------
        model_function : callable
            Model simulation function
        analytical_function : callable
            Analytical solution function
        **kwargs
            Parameters for both functions

        Returns
        -------
        dict
            Comparison results
        """
        # Run model
        model_result = model_function(**kwargs)

        # Get analytical solution
        analytical_result = analytical_function(**kwargs)

        # Calculate error metrics
        if isinstance(model_result, np.ndarray) and isinstance(analytical_result, np.ndarray):
            rmse = np.sqrt(np.mean((model_result - analytical_result)**2))
            mae = np.mean(np.abs(model_result - analytical_result))
            max_error = np.max(np.abs(model_result - analytical_result))

            results = {
                'model': model_result,
                'analytical': analytical_result,
                'rmse': rmse,
                'mae': mae,
                'max_error': max_error,
                'passes': rmse < 0.1  # Arbitrary threshold
            }
        else:
            results = {
                'model': model_result,
                'analytical': analytical_result,
            }

        self.results['infiltration'] = results
        return results

    def run_percolation_benchmark(
        self,
        observed_p_c: float,
        observed_exponents: dict
    ) -> dict:
        """
        Validate percolation threshold and exponents.

        Parameters
        ----------
        observed_p_c : float
            Observed percolation threshold
        observed_exponents : dict
            Observed critical exponents

        Returns
        -------
        dict
            Validation results
        """
        theoretical = percolation_theory_critical_exponents()

        # Determine dimension from exponents
        dim = '2d' if abs(observed_exponents.get('nu', 0) - 4/3) < 0.2 else '3d'

        results = {
            'dimension': dim,
            'p_c_observed': observed_p_c,
            'p_c_theoretical': theoretical[dim]['p_c'],
            'p_c_error': abs(observed_p_c - theoretical[dim]['p_c']),
            'exponents': {}
        }

        # Compare exponents
        for exp_name in ['nu', 'beta', 'gamma', 'mu']:
            if exp_name in observed_exponents:
                obs = observed_exponents[exp_name]
                theo = theoretical[dim][exp_name]
                error = abs(obs - theo) / theo

                results['exponents'][exp_name] = {
                    'observed': obs,
                    'theoretical': theo,
                    'relative_error': error,
                    'passes': error < 0.2  # 20% tolerance
                }

        self.results['percolation'] = results
        return results

    def generate_report(self) -> str:
        """
        Generate validation report.

        Returns
        -------
        str
            Formatted report
        """
        report = "Validation Report\n"
        report += "=" * 60 + "\n\n"

        for test_name, test_results in self.results.items():
            report += f"{test_name.upper()} BENCHMARK\n"
            report += "-" * 40 + "\n"

            if isinstance(test_results, dict):
                for key, value in test_results.items():
                    if isinstance(value, (int, float)):
                        report += f"  {key}: {value:.6f}\n"
                    elif isinstance(value, bool):
                        report += f"  {key}: {'PASS' if value else 'FAIL'}\n"
                    elif isinstance(value, dict):
                        report += f"  {key}:\n"
                        for k, v in value.items():
                            report += f"    {k}: {v}\n"

            report += "\n"

        return report
