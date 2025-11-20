"""
Unit tests for ensemble simulation framework.

Tests cover:
- Parameter distribution
- Parameter sampling
- Ensemble execution
- Statistics calculation
- Latin Hypercube Sampling
- Uncertainty quantification
"""

import numpy as np
import pytest
from soilstocenergy.validation.ensemble import (
    ParameterDistribution,
    EnsembleSimulation,
    latin_hypercube_sampling,
    calculate_sobol_indices,
)


class TestParameterDistribution:
    """Test ParameterDistribution class."""

    def test_initialization(self):
        """Test distribution initialization."""
        dist = ParameterDistribution(
            name='K_sat',
            distribution='normal',
            mean=1e-5,
            std=2e-6,
            bounds=(1e-6, 1e-4)
        )

        assert dist.name == 'K_sat'
        assert dist.distribution == 'normal'
        assert dist.mean == 1e-5
        assert dist.std == 2e-6

    def test_normal_sampling(self):
        """Test normal distribution sampling."""
        dist = ParameterDistribution(
            name='param',
            distribution='normal',
            mean=10.0,
            std=2.0
        )

        samples = dist.sample(n_samples=1000, seed=42)

        # Check mean and std
        assert np.mean(samples) == pytest.approx(10.0, abs=0.3)
        assert np.std(samples) == pytest.approx(2.0, abs=0.3)

    def test_uniform_sampling(self):
        """Test uniform distribution sampling."""
        dist = ParameterDistribution(
            name='param',
            distribution='uniform',
            mean=10.0,
            std=2.0  # Half-width
        )

        samples = dist.sample(n_samples=1000, seed=42)

        # Should be between mean-std and mean+std
        assert np.min(samples) >= 8.0
        assert np.max(samples) <= 12.0

    def test_lognormal_sampling(self):
        """Test lognormal distribution sampling."""
        dist = ParameterDistribution(
            name='param',
            distribution='lognormal',
            mean=1.0,  # Log-space mean
            std=0.5    # Log-space std
        )

        samples = dist.sample(n_samples=1000, seed=42)

        # All should be positive
        assert np.all(samples > 0)

    def test_bounds_application(self):
        """Test that bounds are applied."""
        dist = ParameterDistribution(
            name='param',
            distribution='normal',
            mean=10.0,
            std=5.0,
            bounds=(5.0, 15.0)
        )

        samples = dist.sample(n_samples=1000, seed=42)

        # All samples should be within bounds
        assert np.all(samples >= 5.0)
        assert np.all(samples <= 15.0)

    def test_reproducibility(self):
        """Test that seeding gives reproducible results."""
        dist = ParameterDistribution(
            name='param',
            distribution='normal',
            mean=10.0,
            std=2.0
        )

        samples1 = dist.sample(n_samples=100, seed=42)
        samples2 = dist.sample(n_samples=100, seed=42)

        assert np.allclose(samples1, samples2)

    def test_invalid_distribution(self):
        """Test error handling for invalid distribution."""
        dist = ParameterDistribution(
            name='param',
            distribution='invalid',
            mean=10.0,
            std=2.0
        )

        with pytest.raises(ValueError):
            dist.sample(n_samples=10)


class TestEnsembleSimulation:
    """Test EnsembleSimulation class."""

    def test_initialization(self):
        """Test ensemble initialization."""
        def dummy_model(params):
            return {'result': params['x'] * 2}

        param_dists = [
            ParameterDistribution('x', 'normal', 5.0, 1.0)
        ]

        ensemble = EnsembleSimulation(
            model_function=dummy_model,
            parameter_distributions=param_dists,
            n_ensemble=10
        )

        assert ensemble.n_ensemble == 10
        assert len(ensemble.param_dists) == 1
        assert len(ensemble.results) == 0

    def test_sample_parameters(self):
        """Test parameter sampling."""
        def dummy_model(params):
            return {}

        param_dists = [
            ParameterDistribution('x', 'normal', 5.0, 1.0),
            ParameterDistribution('y', 'uniform', 10.0, 2.0)
        ]

        ensemble = EnsembleSimulation(
            dummy_model, param_dists, n_ensemble=20
        )

        param_samples = ensemble.sample_parameters(seed=42)

        assert len(param_samples) == 20
        assert all('x' in p and 'y' in p for p in param_samples)

    def test_run_ensemble(self):
        """Test running ensemble."""
        def model_function(params):
            return {'output': params['x'] + params['y']}

        param_dists = [
            ParameterDistribution('x', 'normal', 5.0, 1.0),
            ParameterDistribution('y', 'uniform', 10.0, 2.0)
        ]

        ensemble = EnsembleSimulation(
            model_function, param_dists, n_ensemble=10
        )

        ensemble.run_ensemble(seed=42, verbose=False)

        assert len(ensemble.results) == 10
        assert all('output' in r for r in ensemble.results)

    def test_ensemble_reproducibility(self):
        """Test that ensemble is reproducible with seed."""
        def model_function(params):
            return {'output': params['x'] ** 2}

        param_dists = [
            ParameterDistribution('x', 'normal', 5.0, 1.0)
        ]

        ensemble1 = EnsembleSimulation(
            model_function, param_dists, n_ensemble=10
        )
        ensemble1.run_ensemble(seed=42, verbose=False)

        ensemble2 = EnsembleSimulation(
            model_function, param_dists, n_ensemble=10
        )
        ensemble2.run_ensemble(seed=42, verbose=False)

        # Results should be identical
        for r1, r2 in zip(ensemble1.results, ensemble2.results):
            assert r1['output'] == pytest.approx(r2['output'])

    def test_ensemble_with_failure(self):
        """Test ensemble handling of failed members."""
        def model_function(params):
            if params['x'] < 3.0:
                raise ValueError("Parameter out of valid range")
            return {'output': params['x']}

        param_dists = [
            ParameterDistribution('x', 'uniform', 5.0, 3.0, bounds=(0, 10))
        ]

        ensemble = EnsembleSimulation(
            model_function, param_dists, n_ensemble=20
        )

        # Should not raise, but some members may be None
        with pytest.warns(UserWarning):
            ensemble.run_ensemble(seed=42, verbose=False)

        # Some results should be None (failures)
        assert None in ensemble.results


class TestEnsembleStatistics:
    """Test ensemble statistics calculation."""

    def test_get_statistics_scalar(self):
        """Test statistics for scalar variable."""
        def model_function(params):
            return {'result': params['x']}

        param_dists = [
            ParameterDistribution('x', 'normal', 10.0, 2.0)
        ]

        ensemble = EnsembleSimulation(
            model_function, param_dists, n_ensemble=100
        )
        ensemble.run_ensemble(seed=42, verbose=False)

        stats = ensemble.get_statistics('result')

        assert 'mean' in stats
        assert 'std' in stats
        assert 'median' in stats
        assert 'p05' in stats
        assert 'p95' in stats

        # Mean should be close to distribution mean
        assert stats['mean'] == pytest.approx(10.0, abs=0.5)

    def test_get_statistics_array(self):
        """Test statistics for array variable."""
        def model_function(params):
            return {'profile': np.array([params['x'], params['x']*2, params['x']*3])}

        param_dists = [
            ParameterDistribution('x', 'uniform', 5.0, 2.0)
        ]

        ensemble = EnsembleSimulation(
            model_function, param_dists, n_ensemble=50
        )
        ensemble.run_ensemble(seed=42, verbose=False)

        stats = ensemble.get_statistics('profile')

        # Each statistic should be an array
        assert stats['mean'].shape == (3,)
        assert stats['std'].shape == (3,)

        # Mean should scale with multiplier
        assert stats['mean'][1] == pytest.approx(2 * stats['mean'][0], rel=0.1)

    def test_get_statistics_missing_variable(self):
        """Test statistics with missing variable."""
        def model_function(params):
            return {'result': params['x']}

        param_dists = [
            ParameterDistribution('x', 'normal', 5.0, 1.0)
        ]

        ensemble = EnsembleSimulation(
            model_function, param_dists, n_ensemble=10
        )
        ensemble.run_ensemble(seed=42, verbose=False)

        stats = ensemble.get_statistics('nonexistent')

        # Should return empty dict
        assert stats == {}

    def test_percentiles(self):
        """Test percentile calculations."""
        def model_function(params):
            return {'value': params['x']}

        param_dists = [
            ParameterDistribution('x', 'uniform', 10.0, 5.0)  # 5 to 15
        ]

        ensemble = EnsembleSimulation(
            model_function, param_dists, n_ensemble=1000
        )
        ensemble.run_ensemble(seed=42, verbose=False)

        stats = ensemble.get_statistics('value')

        # For uniform distribution
        # p05 should be near 5.5, p95 near 14.5
        assert stats['p05'] < stats['median']
        assert stats['median'] < stats['p95']


class TestPDF:
    """Test probability density function calculation."""

    def test_get_pdf_scalar(self):
        """Test PDF for scalar outputs."""
        def model_function(params):
            return {'x': params['param']}

        param_dists = [
            ParameterDistribution('param', 'normal', 10.0, 2.0)
        ]

        ensemble = EnsembleSimulation(
            model_function, param_dists, n_ensemble=1000
        )
        ensemble.run_ensemble(seed=42, verbose=False)

        hist, bin_centers = ensemble.get_pdf('x')

        # Should have positive counts
        assert np.sum(hist) > 0

        # Peak should be near mean
        peak_value = bin_centers[np.argmax(hist)]
        assert peak_value == pytest.approx(10.0, abs=1.0)

    def test_get_pdf_with_custom_bins(self):
        """Test PDF with custom bins."""
        def model_function(params):
            return {'x': params['param']}

        param_dists = [
            ParameterDistribution('param', 'uniform', 10.0, 5.0)
        ]

        ensemble = EnsembleSimulation(
            model_function, param_dists, n_ensemble=100
        )
        ensemble.run_ensemble(seed=42, verbose=False)

        bins = np.linspace(0, 20, 21)
        hist, bin_centers = ensemble.get_pdf('x', bins=bins)

        assert len(hist) == 20
        assert len(bin_centers) == 20


class TestUncertaintyBands:
    """Test uncertainty band calculation."""

    def test_calculate_uncertainty_bands(self):
        """Test uncertainty band calculation."""
        def model_function(params):
            return {'y': params['x'] ** 2}

        param_dists = [
            ParameterDistribution('x', 'uniform', 5.0, 2.0)
        ]

        ensemble = EnsembleSimulation(
            model_function, param_dists, n_ensemble=100
        )
        ensemble.run_ensemble(seed=42, verbose=False)

        lower, upper = ensemble.calculate_uncertainty_bands('y', confidence=0.90)

        # Upper should be greater than lower
        assert upper > lower

        # Mean should be within bounds
        stats = ensemble.get_statistics('y')
        assert stats['mean'] >= lower
        assert stats['mean'] <= upper

    def test_different_confidence_levels(self):
        """Test different confidence levels."""
        def model_function(params):
            return {'y': params['x']}

        param_dists = [
            ParameterDistribution('x', 'normal', 10.0, 2.0)
        ]

        ensemble = EnsembleSimulation(
            model_function, param_dists, n_ensemble=500
        )
        ensemble.run_ensemble(seed=42, verbose=False)

        lower_90, upper_90 = ensemble.calculate_uncertainty_bands('y', confidence=0.90)
        lower_50, upper_50 = ensemble.calculate_uncertainty_bands('y', confidence=0.50)

        # 90% band should be wider than 50% band
        width_90 = upper_90 - lower_90
        width_50 = upper_50 - lower_50

        assert width_90 > width_50


class TestLatinHypercubeSampling:
    """Test Latin Hypercube Sampling."""

    def test_lhs_basic(self):
        """Test basic LHS functionality."""
        param_dists = [
            ParameterDistribution('x', 'uniform', 10.0, 5.0),
            ParameterDistribution('y', 'uniform', 20.0, 10.0)
        ]

        samples = latin_hypercube_sampling(param_dists, n_samples=50, seed=42)

        assert len(samples) == 50
        assert all('x' in s and 'y' in s for s in samples)

    def test_lhs_coverage(self):
        """Test that LHS provides good parameter space coverage."""
        param_dists = [
            ParameterDistribution('x', 'uniform', 10.0, 5.0)
        ]

        samples = latin_hypercube_sampling(param_dists, n_samples=100, seed=42)

        values = np.array([s['x'] for s in samples])

        # Should span the range well
        assert np.min(values) < 7.0  # Near lower bound
        assert np.max(values) > 13.0  # Near upper bound

    def test_lhs_normal_distribution(self):
        """Test LHS with normal distribution."""
        param_dists = [
            ParameterDistribution('x', 'normal', 10.0, 2.0)
        ]

        samples = latin_hypercube_sampling(param_dists, n_samples=100, seed=42)

        values = np.array([s['x'] for s in samples])

        # Mean should be close to distribution mean
        assert np.mean(values) == pytest.approx(10.0, abs=0.5)

    def test_lhs_lognormal_distribution(self):
        """Test LHS with lognormal distribution."""
        param_dists = [
            ParameterDistribution('x', 'lognormal', 1.0, 0.5)
        ]

        samples = latin_hypercube_sampling(param_dists, n_samples=100, seed=42)

        values = np.array([s['x'] for s in samples])

        # All should be positive
        assert np.all(values > 0)

    def test_lhs_multiple_parameters(self):
        """Test LHS with multiple parameters."""
        param_dists = [
            ParameterDistribution('a', 'uniform', 1.0, 0.5),
            ParameterDistribution('b', 'normal', 5.0, 1.0),
            ParameterDistribution('c', 'uniform', 10.0, 2.0)
        ]

        samples = latin_hypercube_sampling(param_dists, n_samples=50, seed=42)

        assert len(samples) == 50

        # Each parameter should vary
        a_values = [s['a'] for s in samples]
        b_values = [s['b'] for s in samples]
        c_values = [s['c'] for s in samples]

        assert len(set(a_values)) > 40  # Most should be unique
        assert len(set(b_values)) > 40
        assert len(set(c_values)) > 40

    def test_lhs_reproducibility(self):
        """Test LHS reproducibility with seed."""
        param_dists = [
            ParameterDistribution('x', 'uniform', 10.0, 5.0)
        ]

        samples1 = latin_hypercube_sampling(param_dists, n_samples=20, seed=42)
        samples2 = latin_hypercube_sampling(param_dists, n_samples=20, seed=42)

        for s1, s2 in zip(samples1, samples2):
            assert s1['x'] == pytest.approx(s2['x'])

    def test_lhs_respects_bounds(self):
        """Test that LHS respects parameter bounds."""
        param_dists = [
            ParameterDistribution('x', 'normal', 10.0, 5.0, bounds=(0.0, 20.0))
        ]

        samples = latin_hypercube_sampling(param_dists, n_samples=100, seed=42)

        values = np.array([s['x'] for s in samples])

        assert np.all(values >= 0.0)
        assert np.all(values <= 20.0)


class TestSobolIndices:
    """Test Sobol sensitivity indices calculation."""

    def test_sobol_basic(self):
        """Test basic Sobol index calculation."""
        def model_function(params):
            return {'y': params['x1'] + params['x2']}

        param_dists = [
            ParameterDistribution('x1', 'uniform', 5.0, 2.0),
            ParameterDistribution('x2', 'uniform', 10.0, 2.0)
        ]

        # Note: This is a simplified placeholder implementation
        indices = calculate_sobol_indices(model_function, param_dists, n_samples=100)

        # Should return indices for each parameter
        assert 'x1' in indices
        assert 'x2' in indices

        # Indices should sum to approximately 1 (for first-order)
        # Note: Current implementation is placeholder
        total = sum(indices.values())
        assert total == pytest.approx(1.0, abs=0.1)


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_complete_uncertainty_analysis(self):
        """Test complete uncertainty analysis workflow."""
        def soil_water_model(params):
            """Simple soil water model."""
            K = params['K_sat']
            theta = params['theta_initial']
            precip = 1e-5

            # Simplified infiltration
            infiltration = min(precip, K)
            theta_final = theta + infiltration * 3600 / 0.3

            return {
                'infiltration': infiltration,
                'theta_final': theta_final,
                'runoff': max(0, precip - infiltration)
            }

        param_dists = [
            ParameterDistribution('K_sat', 'lognormal', np.log(1e-5), 0.5,
                                bounds=(1e-7, 1e-3)),
            ParameterDistribution('theta_initial', 'uniform', 0.25, 0.1,
                                bounds=(0.05, 0.45))
        ]

        # Run ensemble
        ensemble = EnsembleSimulation(
            soil_water_model, param_dists, n_ensemble=100
        )
        ensemble.run_ensemble(seed=42, verbose=False)

        # Get statistics
        stats_theta = ensemble.get_statistics('theta_final')
        stats_runoff = ensemble.get_statistics('runoff')

        # Should have reasonable statistics
        assert 'mean' in stats_theta
        assert 'p05' in stats_theta
        assert 'p95' in stats_theta

        # Uncertainty bands
        lower, upper = ensemble.calculate_uncertainty_bands('theta_final', confidence=0.90)

        assert upper > lower
        assert stats_theta['mean'] >= lower
        assert stats_theta['mean'] <= upper

    def test_ensemble_vs_lhs(self):
        """Test comparing random ensemble with LHS."""
        def simple_model(params):
            return {'output': params['x'] ** 2 + params['y']}

        param_dists = [
            ParameterDistribution('x', 'uniform', 5.0, 2.0),
            ParameterDistribution('y', 'uniform', 10.0, 2.0)
        ]

        # Random sampling
        ensemble_random = EnsembleSimulation(
            simple_model, param_dists, n_ensemble=50
        )
        ensemble_random.run_ensemble(seed=42, verbose=False)

        # LHS - simulate by manually sampling
        lhs_samples = latin_hypercube_sampling(param_dists, n_samples=50, seed=42)
        results_lhs = [simple_model(params) for params in lhs_samples]

        # Both should give similar statistics
        # (With more samples, LHS typically more efficient)
        mean_random = np.mean([r['output'] for r in ensemble_random.results])
        mean_lhs = np.mean([r['output'] for r in results_lhs])

        # Should be in same ballpark
        assert abs(mean_random - mean_lhs) < 5.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
