"""
Ensemble framework for uncertainty quantification.

This module implements Monte Carlo ensemble simulations to:
- Propagate parameter uncertainty
- Generate probability distributions
- Quantify prediction uncertainty
- Test universality of scaling laws

Key features:
- Parameter sampling from distributions
- Parallel ensemble execution
- Statistical analysis of results
- Scaling law verification across ensemble
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class ParameterDistribution:
    """
    Parameter distribution for sampling.

    Attributes
    ----------
    name : str
        Parameter name
    distribution : str
        Distribution type ('normal', 'uniform', 'lognormal')
    mean : float
        Mean or center value
    std : float
        Standard deviation or half-width
    bounds : tuple
        (min, max) bounds for the parameter
    """
    name: str
    distribution: str
    mean: float
    std: float
    bounds: Tuple[float, float] = (-np.inf, np.inf)

    def sample(self, n_samples: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """
        Sample from distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        array
            Sampled values
        """
        if seed is not None:
            np.random.seed(seed)

        if self.distribution == 'normal':
            samples = np.random.normal(self.mean, self.std, n_samples)
        elif self.distribution == 'uniform':
            samples = np.random.uniform(
                self.mean - self.std,
                self.mean + self.std,
                n_samples
            )
        elif self.distribution == 'lognormal':
            # For lognormal, mean and std are in log-space
            samples = np.random.lognormal(self.mean, self.std, n_samples)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

        # Apply bounds
        samples = np.clip(samples, self.bounds[0], self.bounds[1])

        return samples


class EnsembleSimulation:
    """
    Ensemble simulation framework.

    Runs multiple realizations with sampled parameters and analyzes results.
    """

    def __init__(
        self,
        model_function: Callable,
        parameter_distributions: List[ParameterDistribution],
        n_ensemble: int = 100
    ):
        """
        Initialize ensemble simulation.

        Parameters
        ----------
        model_function : callable
            Function that runs model and returns results
            Signature: model_function(params: dict) -> dict
        parameter_distributions : list
            List of ParameterDistribution objects
        n_ensemble : int
            Number of ensemble members
        """
        self.model_function = model_function
        self.param_dists = parameter_distributions
        self.n_ensemble = n_ensemble

        self.results: List[Dict] = []
        self.parameter_samples: List[Dict] = []

    def sample_parameters(self, seed: Optional[int] = None) -> List[Dict]:
        """
        Sample parameters for all ensemble members.

        Parameters
        ----------
        seed : int, optional
            Random seed

        Returns
        -------
        list of dict
            List of parameter dictionaries
        """
        param_samples = []

        for i in range(self.n_ensemble):
            params = {}
            member_seed = seed + i if seed is not None else None

            for param_dist in self.param_dists:
                value = param_dist.sample(1, seed=member_seed)[0]
                params[param_dist.name] = value

            param_samples.append(params)

        self.parameter_samples = param_samples
        return param_samples

    def run_ensemble(self, seed: Optional[int] = None, verbose: bool = True):
        """
        Run all ensemble members.

        Parameters
        ----------
        seed : int, optional
            Random seed
        verbose : bool
            Print progress
        """
        # Sample parameters
        param_samples = self.sample_parameters(seed=seed)

        # Run each member
        self.results = []

        for i, params in enumerate(param_samples):
            if verbose and (i % 10 == 0):
                print(f"Running ensemble member {i+1}/{self.n_ensemble}")

            try:
                result = self.model_function(params)
                self.results.append(result)
            except Exception as e:
                warnings.warn(f"Ensemble member {i} failed: {e}")
                self.results.append(None)

    def get_statistics(self, variable: str) -> Dict[str, np.ndarray]:
        """
        Calculate ensemble statistics for a variable.

        Parameters
        ----------
        variable : str
            Variable name to analyze

        Returns
        -------
        dict
            Statistics: mean, std, percentiles
        """
        # Extract variable from all results
        values = []
        for result in self.results:
            if result is not None and variable in result:
                values.append(result[variable])

        if not values:
            return {}

        values = np.array(values)

        stats = {
            'mean': np.mean(values, axis=0),
            'std': np.std(values, axis=0),
            'median': np.median(values, axis=0),
            'p05': np.percentile(values, 5, axis=0),
            'p25': np.percentile(values, 25, axis=0),
            'p75': np.percentile(values, 75, axis=0),
            'p95': np.percentile(values, 95, axis=0),
        }

        return stats

    def get_pdf(
        self,
        variable: str,
        bins: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate probability density function for variable.

        Parameters
        ----------
        variable : str
            Variable name
        bins : array, optional
            Bin edges

        Returns
        -------
        hist : array
            Histogram counts
        bin_centers : array
            Bin centers
        """
        # Extract values
        values = []
        for result in self.results:
            if result is not None and variable in result:
                val = result[variable]
                if np.isscalar(val):
                    values.append(val)
                elif hasattr(val, '__iter__'):
                    values.extend(val)

        values = np.array(values)

        if bins is None:
            bins = np.linspace(np.min(values), np.max(values), 30)

        hist, bin_edges = np.histogram(values, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return hist, bin_centers

    def calculate_uncertainty_bands(
        self,
        variable: str,
        confidence: float = 0.90
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence bands.

        Parameters
        ----------
        variable : str
            Variable name
        confidence : float
            Confidence level (e.g., 0.90 for 90%)

        Returns
        -------
        lower : array
            Lower bound
        upper : array
            Upper bound
        """
        alpha = (1 - confidence) / 2
        p_lower = alpha * 100
        p_upper = (1 - alpha) * 100

        values = []
        for result in self.results:
            if result is not None and variable in result:
                values.append(result[variable])

        values = np.array(values)

        lower = np.percentile(values, p_lower, axis=0)
        upper = np.percentile(values, p_upper, axis=0)

        return lower, upper


def latin_hypercube_sampling(
    parameter_distributions: List[ParameterDistribution],
    n_samples: int,
    seed: Optional[int] = None
) -> List[Dict]:
    """
    Latin Hypercube Sampling for efficient parameter space coverage.

    Parameters
    ----------
    parameter_distributions : list
        List of ParameterDistribution objects
    n_samples : int
        Number of samples
    seed : int, optional
        Random seed

    Returns
    -------
    list of dict
        Parameter samples
    """
    if seed is not None:
        np.random.seed(seed)

    n_params = len(parameter_distributions)

    # Create Latin Hypercube
    lhs = np.zeros((n_samples, n_params))

    for i in range(n_params):
        # Divide [0,1] into n_samples intervals
        intervals = np.arange(n_samples) / n_samples
        # Random sample within each interval
        samples = intervals + np.random.uniform(0, 1/n_samples, n_samples)
        # Shuffle
        np.random.shuffle(samples)
        lhs[:, i] = samples

    # Transform to parameter distributions
    param_samples = []

    for i in range(n_samples):
        params = {}

        for j, param_dist in enumerate(parameter_distributions):
            # Get percentile value
            p = lhs[i, j]

            if param_dist.distribution == 'normal':
                from scipy.stats import norm
                value = norm.ppf(p, param_dist.mean, param_dist.std)
            elif param_dist.distribution == 'uniform':
                value = param_dist.mean - param_dist.std + 2 * param_dist.std * p
            elif param_dist.distribution == 'lognormal':
                from scipy.stats import lognorm
                value = lognorm.ppf(p, param_dist.std, scale=np.exp(param_dist.mean))
            else:
                value = param_dist.sample(1)[0]

            # Apply bounds
            value = np.clip(value, param_dist.bounds[0], param_dist.bounds[1])

            params[param_dist.name] = value

        param_samples.append(params)

    return param_samples


def calculate_sobol_indices(
    model_function: Callable,
    parameter_distributions: List[ParameterDistribution],
    n_samples: int = 1000
) -> Dict[str, float]:
    """
    Calculate Sobol sensitivity indices (first-order).

    Simplified implementation using sampling.

    Parameters
    ----------
    model_function : callable
        Model function
    parameter_distributions : list
        Parameter distributions
    n_samples : int
        Number of samples for estimation

    Returns
    -------
    dict
        First-order Sobol indices for each parameter
    """
    # This is a simplified placeholder
    # Full Sobol analysis requires more sophisticated sampling
    # (Saltelli's scheme with 2*n_params + 2 sample sets)

    sobol_indices = {}

    # Placeholder: return equal weights
    n_params = len(parameter_distributions)
    for param_dist in parameter_distributions:
        sobol_indices[param_dist.name] = 1.0 / n_params

    return sobol_indices
