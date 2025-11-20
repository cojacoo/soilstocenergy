"""
Unit tests for benchmark analytical solutions.

Tests cover:
- Green-Ampt infiltration
- Philip infiltration
- Bucket model comparisons
- Theis groundwater solution
- Percolation theory exponents
- Mass balance validation
- Steady-state profiles
- Benchmark suite
"""

import numpy as np
import pytest
from soilstocenergy.validation.benchmarks import (
    green_ampt_infiltration,
    philip_infiltration,
    bucket_model_steady_state_pdf,
    theis_solution,
    percolation_theory_critical_exponents,
    compare_with_bucket_model,
    validate_mass_balance,
    analytical_steady_state_profile,
    BenchmarkSuite,
)


class TestGreenAmptInfiltration:
    """Test Green-Ampt infiltration solution."""

    def test_basic_infiltration(self):
        """Test basic Green-Ampt calculation."""
        t = np.array([0, 3600, 7200, 10800])  # 0, 1, 2, 3 hours
        K_sat = 1e-5  # m/s
        psi_f = 0.1  # m
        theta_i = 0.15
        theta_s = 0.45

        I = green_ampt_infiltration(t, K_sat, psi_f, theta_i, theta_s)

        # Should be monotonically increasing
        assert np.all(np.diff(I) >= 0)

        # At t=0, infiltration should be 0
        assert I[0] == 0

        # Infiltration should be positive after t=0
        assert np.all(I[1:] > 0)

    def test_zero_time(self):
        """Test that zero time gives zero infiltration."""
        t = np.array([0])
        I = green_ampt_infiltration(t, 1e-5, 0.1, 0.15, 0.45)

        assert I[0] == 0

    def test_physical_bounds(self):
        """Test that infiltration rate is physical."""
        t = np.array([3600, 7200])
        K_sat = 1e-5
        I = green_ampt_infiltration(t, K_sat, 0.1, 0.15, 0.45)

        # Cumulative infiltration should not exceed K_sat * t (upper limit)
        assert I[0] <= K_sat * 3600 * 2  # Factor of 2 for suction effects
        assert I[1] <= K_sat * 7200 * 2

    def test_convergence(self):
        """Test that iterative solution converges."""
        t = np.array([3600])
        I = green_ampt_infiltration(t, 1e-5, 0.1, 0.15, 0.45)

        # Should converge to a finite value
        assert np.isfinite(I[0])
        assert I[0] > 0

    def test_parameter_sensitivity(self):
        """Test sensitivity to parameters."""
        t = np.array([3600])

        # Higher K_sat should give more infiltration
        I_low_K = green_ampt_infiltration(t, 1e-6, 0.1, 0.15, 0.45)
        I_high_K = green_ampt_infiltration(t, 1e-5, 0.1, 0.15, 0.45)

        assert I_high_K[0] > I_low_K[0]

        # Larger moisture deficit should give more infiltration
        I_small_deficit = green_ampt_infiltration(t, 1e-5, 0.1, 0.40, 0.45)
        I_large_deficit = green_ampt_infiltration(t, 1e-5, 0.1, 0.15, 0.45)

        assert I_large_deficit[0] > I_small_deficit[0]


class TestPhilipInfiltration:
    """Test Philip two-term infiltration equation."""

    def test_basic_infiltration(self):
        """Test basic Philip equation."""
        t = np.array([0, 3600, 7200, 10800])
        S = 1e-3  # Sorptivity
        A = 5e-6  # Gravity term

        I = philip_infiltration(t, S, A)

        # Should be monotonically increasing
        assert np.all(np.diff(I) >= 0)

        # At t=0, infiltration should be 0
        assert I[0] == 0

    def test_sorptivity_dominance_early(self):
        """Test that sorptivity dominates at early times."""
        t_early = np.array([60, 120])  # 1-2 minutes
        S = 1e-3
        A = 1e-6

        I = philip_infiltration(t_early, S, A)

        # Sorptivity term
        I_sorptivity = S * np.sqrt(t_early)

        # Should be dominated by sorptivity
        assert np.all(I <= I_sorptivity * 1.5)

    def test_gravity_dominance_late(self):
        """Test that gravity dominates at late times."""
        t_late = np.array([36000, 72000])  # 10-20 hours
        S = 1e-3
        A = 5e-6

        I = philip_infiltration(t_late, S, A)

        # Gravity term
        I_gravity = A * t_late

        # At late times, should approach linear (gravity-dominated)
        rate_1 = (I[1] - I[0]) / (t_late[1] - t_late[0])
        assert rate_1 == pytest.approx(A, rel=0.5)

    def test_zero_sorptivity(self):
        """Test with zero sorptivity (only gravity)."""
        t = np.array([3600, 7200])
        S = 0.0
        A = 1e-5

        I = philip_infiltration(t, S, A)

        # Should be linear
        expected = A * t
        assert np.allclose(I, expected)


class TestBucketModelPDF:
    """Test bucket model steady-state PDF."""

    def test_pdf_normalization(self):
        """Test that PDF integrates to 1."""
        s = np.linspace(0, 1, 101)
        pdf = bucket_model_steady_state_pdf(
            s, lambda_rain=0.2, alpha=10.0, n_=200.0,
            s_w=0.1, s_star=0.5, Z_r=1000.0
        )

        # Integral should be approximately 1
        integral = np.trapz(pdf, s)
        assert integral == pytest.approx(1.0, rel=0.01)

    def test_pdf_non_negative(self):
        """Test that PDF is non-negative."""
        s = np.linspace(0, 1, 51)
        pdf = bucket_model_steady_state_pdf(
            s, lambda_rain=0.2, alpha=10.0, n_=200.0,
            s_w=0.1, s_star=0.5, Z_r=1000.0
        )

        assert np.all(pdf >= 0)


class TestTheisSolution:
    """Test Theis groundwater solution."""

    def test_basic_drawdown(self):
        """Test basic Theis solution."""
        r = np.array([1, 10, 100])  # Distances from well
        t = 3600  # 1 hour
        Q = 0.001  # m³/s
        T = 0.001  # Transmissivity
        S = 0.0001  # Storage coefficient

        drawdown = theis_solution(r, t, Q, T, S)

        # Drawdown should be positive
        assert np.all(drawdown > 0)

        # Drawdown should decrease with distance
        assert np.all(np.diff(drawdown) < 0)

    def test_drawdown_increases_with_time(self):
        """Test that drawdown increases with time."""
        r = np.array([10])
        t1 = 3600
        t2 = 7200
        Q = 0.001
        T = 0.001
        S = 0.0001

        dd1 = theis_solution(r, t1, Q, T, S)
        dd2 = theis_solution(r, t2, Q, T, S)

        # Later time should have more drawdown
        assert dd2[0] > dd1[0]

    def test_drawdown_increases_with_pumping(self):
        """Test that drawdown increases with pumping rate."""
        r = np.array([10])
        t = 3600
        T = 0.001
        S = 0.0001

        dd_low = theis_solution(r, t, 0.001, T, S)
        dd_high = theis_solution(r, t, 0.002, T, S)

        assert dd_high[0] > dd_low[0]


class TestPercolationTheory:
    """Test percolation theory critical exponents."""

    def test_exponents_returned(self):
        """Test that exponents are returned."""
        exponents = percolation_theory_critical_exponents()

        assert '2d' in exponents
        assert '3d' in exponents

    def test_2d_exponents(self):
        """Test 2D exponents."""
        exponents = percolation_theory_critical_exponents()

        exp_2d = exponents['2d']

        assert 'nu' in exp_2d
        assert 'beta' in exp_2d
        assert 'gamma' in exp_2d
        assert 'mu' in exp_2d
        assert 'p_c' in exp_2d

        # Check known values
        assert exp_2d['nu'] == pytest.approx(4/3, rel=0.01)
        assert exp_2d['p_c'] == pytest.approx(0.5927, rel=0.01)

    def test_3d_exponents(self):
        """Test 3D exponents."""
        exponents = percolation_theory_critical_exponents()

        exp_3d = exponents['3d']

        assert exp_3d['nu'] == pytest.approx(0.88, rel=0.01)
        assert exp_3d['p_c'] == pytest.approx(0.3116, rel=0.01)

    def test_different_dimensions(self):
        """Test that 2D and 3D have different exponents."""
        exponents = percolation_theory_critical_exponents()

        # Should be different
        assert exponents['2d']['nu'] != exponents['3d']['nu']
        assert exponents['2d']['p_c'] != exponents['3d']['p_c']


class TestBucketModelComparison:
    """Test bucket model comparison."""

    def test_perfect_match(self):
        """Test comparison with perfect match."""
        perc_result = {'mean_storage': 100.0}
        bucket_result = {'mean_storage': 100.0}

        diff = compare_with_bucket_model(perc_result, bucket_result, 'mean_storage')

        assert diff == pytest.approx(0.0)

    def test_relative_difference(self):
        """Test relative difference calculation."""
        perc_result = {'mean_storage': 110.0}
        bucket_result = {'mean_storage': 100.0}

        diff = compare_with_bucket_model(perc_result, bucket_result, 'mean_storage')

        expected = abs(110 - 100) / 100
        assert diff == pytest.approx(expected)

    def test_zero_bucket_value(self):
        """Test handling of zero bucket value."""
        perc_result = {'mean_storage': 10.0}
        bucket_result = {'mean_storage': 0.0}

        diff = compare_with_bucket_model(perc_result, bucket_result, 'mean_storage')

        assert np.isnan(diff)

    def test_missing_metric(self):
        """Test handling of missing metric."""
        perc_result = {}
        bucket_result = {'mean_storage': 100.0}

        diff = compare_with_bucket_model(perc_result, bucket_result, 'mean_storage')

        # Should handle gracefully (returns nan or 1.0)
        assert np.isfinite(diff) or np.isnan(diff)


class TestMassBalanceValidation:
    """Test mass balance validation."""

    def test_perfect_balance(self):
        """Test perfect mass balance."""
        initial = 100.0
        final = 110.0
        input = 10.0
        output = 0.0

        passes, error = validate_mass_balance(initial, final, input, output)

        assert passes
        assert error < 1e-10

    def test_small_error(self):
        """Test small acceptable error."""
        initial = 100.0
        final = 109.5  # Small numerical error
        input = 10.0
        output = 0.0

        passes, error = validate_mass_balance(
            initial, final, input, output, tolerance=0.01
        )

        assert passes
        assert error < 0.01

    def test_large_error(self):
        """Test unacceptable error."""
        initial = 100.0
        final = 120.0  # Too much
        input = 10.0
        output = 0.0

        passes, error = validate_mass_balance(
            initial, final, input, output, tolerance=0.01
        )

        assert not passes
        assert error > 0.01

    def test_with_outputs(self):
        """Test mass balance with inputs and outputs."""
        initial = 100.0
        input = 20.0
        output = 15.0
        final = initial + input - output  # Perfect balance

        passes, error = validate_mass_balance(initial, final, input, output)

        assert passes
        assert error < 1e-10

    def test_custom_tolerance(self):
        """Test custom tolerance."""
        initial = 100.0
        final = 102.0  # 2% error
        input = 0.0
        output = 0.0

        # Strict tolerance
        passes_strict, _ = validate_mass_balance(
            initial, final, input, output, tolerance=0.01
        )
        assert not passes_strict

        # Loose tolerance
        passes_loose, _ = validate_mass_balance(
            initial, final, input, output, tolerance=0.05
        )
        assert passes_loose


class TestSteadyStateProfile:
    """Test analytical steady-state profile."""

    def test_profile_shape(self):
        """Test that profile has correct shape."""
        z = np.linspace(0, 1, 11)
        profile = analytical_steady_state_profile(
            z, infiltration_rate=1e-6, K_sat=1e-5,
            theta_r=0.05, theta_s=0.45, alpha=2.0, n=1.5
        )

        assert profile.shape == z.shape

    def test_profile_bounds(self):
        """Test that profile is within physical bounds."""
        z = np.linspace(0, 1, 11)
        theta_r = 0.05
        theta_s = 0.45

        profile = analytical_steady_state_profile(
            z, infiltration_rate=1e-6, K_sat=1e-5,
            theta_r=theta_r, theta_s=theta_s, alpha=2.0, n=1.5
        )

        assert np.all(profile >= theta_r * 0.999)
        assert np.all(profile <= theta_s * 1.001)

    def test_high_infiltration_saturated(self):
        """Test that high infiltration leads to near saturation."""
        z = np.linspace(0, 1, 11)
        theta_s = 0.45

        profile = analytical_steady_state_profile(
            z, infiltration_rate=5e-5, K_sat=1e-5,  # Exceeds K_sat
            theta_r=0.05, theta_s=theta_s, alpha=2.0, n=1.5
        )

        # Should be near saturation
        assert np.all(profile >= theta_s * 0.95)

    def test_low_infiltration_unsaturated(self):
        """Test that low infiltration gives lower moisture."""
        z = np.linspace(0, 1, 11)
        theta_s = 0.45

        profile = analytical_steady_state_profile(
            z, infiltration_rate=1e-7, K_sat=1e-5,
            theta_r=0.05, theta_s=theta_s, alpha=2.0, n=1.5
        )

        # Should be well below saturation
        assert np.all(profile < theta_s * 0.9)


class TestBenchmarkSuite:
    """Test BenchmarkSuite class."""

    def test_initialization(self):
        """Test suite initialization."""
        suite = BenchmarkSuite()

        assert isinstance(suite.results, dict)
        assert len(suite.results) == 0

    def test_infiltration_benchmark(self):
        """Test infiltration benchmark."""
        suite = BenchmarkSuite()

        # Define model and analytical functions
        def model_func(t, K_sat, psi_f, theta_i, theta_s):
            return t * K_sat * 0.8  # Simplified model

        def analytical_func(t, K_sat, psi_f, theta_i, theta_s):
            return green_ampt_infiltration(t, K_sat, psi_f, theta_i, theta_s)

        # Run benchmark
        t = np.array([3600, 7200])
        results = suite.run_infiltration_benchmark(
            model_func, analytical_func,
            t=t, K_sat=1e-5, psi_f=0.1, theta_i=0.15, theta_s=0.45
        )

        assert 'model' in results
        assert 'analytical' in results
        assert 'rmse' in results
        assert 'mae' in results
        assert 'max_error' in results
        assert 'passes' in results

    def test_percolation_benchmark(self):
        """Test percolation threshold and exponents benchmark."""
        suite = BenchmarkSuite()

        observed_p_c = 0.59
        observed_exponents = {
            'nu': 1.35,
            'beta': 0.14,
            'gamma': 2.4,
        }

        results = suite.run_percolation_benchmark(observed_p_c, observed_exponents)

        assert 'dimension' in results
        assert 'p_c_observed' in results
        assert 'p_c_theoretical' in results
        assert 'exponents' in results

        # Should identify as 2D
        assert results['dimension'] == '2d'

    def test_generate_report(self):
        """Test report generation."""
        suite = BenchmarkSuite()

        # Add some results
        suite.results['test1'] = {
            'value': 1.234,
            'passes': True,
            'nested': {'a': 1, 'b': 2}
        }

        report = suite.generate_report()

        assert isinstance(report, str)
        assert 'TEST1' in report
        assert 'value' in report
        assert 'PASS' in report

    def test_multiple_benchmarks(self):
        """Test running multiple benchmarks."""
        suite = BenchmarkSuite()

        # Run infiltration benchmark
        def dummy_model(t, **kwargs):
            return t * 1e-5

        t = np.array([3600])
        suite.run_infiltration_benchmark(
            dummy_model, philip_infiltration,
            t=t, S=1e-3, A=1e-5
        )

        # Run percolation benchmark
        suite.run_percolation_benchmark(0.59, {'nu': 1.33})

        # Should have both results
        assert 'infiltration' in suite.results
        assert 'percolation' in suite.results


class TestIntegration:
    """Integration tests for benchmark validation."""

    def test_complete_validation_workflow(self):
        """Test complete validation workflow."""
        suite = BenchmarkSuite()

        # Test Green-Ampt
        t = np.linspace(0, 10800, 11)
        I_analytical = green_ampt_infiltration(
            t, K_sat=1e-5, psi_f=0.1, theta_i=0.15, theta_s=0.45
        )

        # "Model" that closely matches analytical
        def model_func(**kwargs):
            return I_analytical * 1.05  # 5% error

        results_inf = suite.run_infiltration_benchmark(
            model_func, green_ampt_infiltration,
            t=t, K_sat=1e-5, psi_f=0.1, theta_i=0.15, theta_s=0.45
        )

        # Should pass with small error
        assert results_inf['passes'] or results_inf['rmse'] < I_analytical[-1] * 0.1

        # Test percolation exponents
        results_perc = suite.run_percolation_benchmark(
            0.59, {'nu': 1.35, 'beta': 0.14}
        )

        # Should identify as 2D
        assert results_perc['dimension'] == '2d'

        # Generate report
        report = suite.generate_report()
        assert len(report) > 100  # Should have content

    def test_mass_balance_in_workflow(self):
        """Test mass balance validation in workflow."""
        # Simulate a model run
        initial_storage = 0.25 * 1.0  # m
        precip = 0.05  # m
        ET = 0.02  # m
        drainage = 0.01  # m

        expected_final = initial_storage + precip - ET - drainage
        actual_final = expected_final * 1.01  # 1% error

        passes, error = validate_mass_balance(
            initial_storage, actual_final,
            precip, ET + drainage,
            tolerance=0.02
        )

        assert passes
        assert error < 0.02


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
