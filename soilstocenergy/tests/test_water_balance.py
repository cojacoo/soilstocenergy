"""
Unit tests for water balance utilities.

Tests cover:
- Storage change calculations
- Connectivity-aware water balance
- Trapped water calculations
- Mobile water calculations
"""

import numpy as np
import pytest
from soilstocenergy.core.water_balance import (
    calculate_storage_change,
    calculate_trapped_water,
    calculate_mobile_water,
)


class TestStorageChange:
    """Test storage change calculations."""

    def test_basic_storage_change(self):
        """Test basic storage change without connectivity."""
        theta = np.array([0.30, 0.25, 0.20])
        flux_in = np.array([1e-5, 1e-5, 1e-5])
        flux_out = np.array([8e-6, 8e-6, 8e-6])
        ET = np.array([2e-6, 2e-6, 2e-6])
        thickness = 0.3
        dt = 3600  # 1 hour
        kappa = np.ones(3)  # All connected

        theta_new = calculate_storage_change(
            theta, flux_in, flux_out, ET, thickness, dt, kappa
        )

        # All should increase (flux_in > flux_out + ET)
        assert np.all(theta_new > theta)

    def test_connected_vs_disconnected(self):
        """Test that disconnected elements only lose water via ET."""
        theta = np.array([0.30, 0.30])
        flux_in = np.array([1e-5, 1e-5])
        flux_out = np.array([5e-6, 5e-6])
        ET = np.array([2e-6, 2e-6])
        thickness = 0.3
        dt = 3600

        # First connected, second disconnected
        kappa = np.array([0.9, 0.1])

        theta_new = calculate_storage_change(
            theta, flux_in, flux_out, ET, thickness, dt, kappa
        )

        # Connected element: flux_in - flux_out - ET = 1e-5 - 5e-6 - 2e-6 = 3e-6 > 0
        assert theta_new[0] > theta[0]

        # Disconnected element: only ET loss
        # Change = -ET/thickness * dt = -2e-6/0.3 * 3600
        expected_change = -ET[1] * dt / thickness
        assert theta_new[1] == pytest.approx(theta[1] + expected_change, rel=1e-6)
        assert theta_new[1] < theta[1]

    def test_no_flux_no_ET_connected(self):
        """Test that connected elements with no fluxes stay constant."""
        theta = np.array([0.25])
        flux_in = np.array([0.0])
        flux_out = np.array([0.0])
        ET = np.array([0.0])
        thickness = 0.3
        dt = 3600
        kappa = np.array([1.0])

        theta_new = calculate_storage_change(
            theta, flux_in, flux_out, ET, thickness, dt, kappa
        )

        assert theta_new[0] == pytest.approx(theta[0], abs=1e-10)

    def test_no_flux_no_ET_disconnected(self):
        """Test that disconnected elements with no fluxes stay constant."""
        theta = np.array([0.25])
        flux_in = np.array([0.0])
        flux_out = np.array([0.0])
        ET = np.array([0.0])
        thickness = 0.3
        dt = 3600
        kappa = np.array([0.0])

        theta_new = calculate_storage_change(
            theta, flux_in, flux_out, ET, thickness, dt, kappa
        )

        assert theta_new[0] == pytest.approx(theta[0], abs=1e-10)

    def test_connectivity_threshold(self):
        """Test connectivity threshold parameter."""
        theta = np.array([0.30, 0.30])
        flux_in = np.array([1e-5, 1e-5])
        flux_out = np.array([5e-6, 5e-6])
        ET = np.array([2e-6, 2e-6])
        thickness = 0.3
        dt = 3600

        # Both at threshold (0.5)
        kappa = np.array([0.51, 0.49])

        theta_new = calculate_storage_change(
            theta, flux_in, flux_out, ET, thickness, dt, kappa,
            connectivity_threshold=0.5
        )

        # First should be treated as connected
        assert theta_new[0] > theta[0]
        # Second should be treated as disconnected
        assert theta_new[1] < theta[1]

    def test_array_shapes(self):
        """Test that function handles various array shapes."""
        for n in [1, 5, 10, 100]:
            theta = np.full(n, 0.25)
            flux_in = np.full(n, 1e-5)
            flux_out = np.full(n, 5e-6)
            ET = np.full(n, 2e-6)
            kappa = np.ones(n)

            theta_new = calculate_storage_change(
                theta, flux_in, flux_out, ET, 0.3, 3600, kappa
            )

            assert theta_new.shape == (n,)


class TestTrappedWater:
    """Test trapped water calculations."""

    def test_all_connected(self):
        """Test trapped water when all elements connected."""
        theta = np.array([0.30, 0.25, 0.20])
        kappa = np.array([1.0, 0.9, 0.8])  # All connected
        thickness = np.array([0.3, 0.3, 0.3])

        trapped = calculate_trapped_water(theta, kappa, thickness)

        # No trapped water when all connected
        assert trapped == pytest.approx(0.0)

    def test_all_disconnected(self):
        """Test trapped water when all elements disconnected."""
        theta = np.array([0.30, 0.25, 0.20])
        kappa = np.array([0.1, 0.2, 0.3])  # All disconnected
        thickness = np.array([0.3, 0.3, 0.3])

        trapped = calculate_trapped_water(theta, kappa, thickness)

        # All water is trapped
        expected = np.sum(theta * thickness)
        assert trapped == pytest.approx(expected)

    def test_mixed_connectivity(self):
        """Test trapped water with mixed connectivity."""
        theta = np.array([0.30, 0.25, 0.20, 0.15])
        kappa = np.array([0.9, 0.3, 0.8, 0.2])  # Alternating
        thickness = np.array([0.3, 0.3, 0.3, 0.3])

        trapped = calculate_trapped_water(theta, kappa, thickness)

        # Only elements at index 1 and 3 are disconnected
        expected = theta[1] * thickness[1] + theta[3] * thickness[3]
        assert trapped == pytest.approx(expected)

    def test_custom_threshold(self):
        """Test trapped water with custom threshold."""
        theta = np.array([0.30, 0.25])
        kappa = np.array([0.6, 0.4])
        thickness = np.array([0.3, 0.3])

        # Threshold 0.5
        trapped_default = calculate_trapped_water(
            theta, kappa, thickness, connectivity_threshold=0.5
        )
        expected_default = theta[1] * thickness[1]  # Only second element trapped
        assert trapped_default == pytest.approx(expected_default)

        # Threshold 0.7
        trapped_high = calculate_trapped_water(
            theta, kappa, thickness, connectivity_threshold=0.7
        )
        expected_high = np.sum(theta * thickness)  # Both trapped
        assert trapped_high == pytest.approx(expected_high)


class TestMobileWater:
    """Test mobile water calculations."""

    def test_all_connected(self):
        """Test mobile water when all elements connected."""
        theta = np.array([0.30, 0.25, 0.20])
        kappa = np.array([1.0, 0.9, 0.8])
        thickness = np.array([0.3, 0.3, 0.3])

        mobile = calculate_mobile_water(theta, kappa, thickness)

        # All water is mobile
        expected = np.sum(theta * thickness)
        assert mobile == pytest.approx(expected)

    def test_all_disconnected(self):
        """Test mobile water when all elements disconnected."""
        theta = np.array([0.30, 0.25, 0.20])
        kappa = np.array([0.1, 0.2, 0.3])
        thickness = np.array([0.3, 0.3, 0.3])

        mobile = calculate_mobile_water(theta, kappa, thickness)

        # No mobile water
        assert mobile == pytest.approx(0.0)

    def test_mixed_connectivity(self):
        """Test mobile water with mixed connectivity."""
        theta = np.array([0.30, 0.25, 0.20, 0.15])
        kappa = np.array([0.9, 0.3, 0.8, 0.2])
        thickness = np.array([0.3, 0.3, 0.3, 0.3])

        mobile = calculate_mobile_water(theta, kappa, thickness)

        # Only elements at index 0 and 2 are connected
        expected = theta[0] * thickness[0] + theta[2] * thickness[2]
        assert mobile == pytest.approx(expected)

    def test_mobile_plus_trapped_equals_total(self):
        """Test that mobile + trapped equals total water."""
        theta = np.array([0.30, 0.25, 0.20, 0.15, 0.18])
        kappa = np.array([0.9, 0.3, 0.8, 0.2, 0.6])
        thickness = np.array([0.3, 0.3, 0.3, 0.3, 0.3])

        mobile = calculate_mobile_water(theta, kappa, thickness)
        trapped = calculate_trapped_water(theta, kappa, thickness)
        total = np.sum(theta * thickness)

        assert mobile + trapped == pytest.approx(total)


class TestIntegration:
    """Integration tests combining multiple water balance components."""

    def test_full_water_balance_cycle(self):
        """Test complete water balance cycle."""
        # Initialize
        n_layers = 5
        theta = np.full(n_layers, 0.20)
        thickness_arr = np.full(n_layers, 0.2)
        dt = 3600

        # All connected initially
        kappa = np.ones(n_layers)

        # Precipitation at top
        precip_rate = 1e-5

        # ET from all layers
        ET = np.full(n_layers, 1e-6)

        # Initial storage
        storage_initial = np.sum(theta * thickness_arr)

        # Apply top boundary (infiltration)
        flux_in = np.zeros(n_layers)
        flux_in[0] = precip_rate

        # No lateral fluxes for this test
        flux_out = np.zeros(n_layers)

        # Update storage
        theta_new = calculate_storage_change(
            theta, flux_in, flux_out, ET, thickness_arr, dt, kappa
        )

        # Final storage
        storage_final = np.sum(theta_new * thickness_arr)

        # Net input should increase storage
        net_input = precip_rate - np.sum(ET)
        assert net_input > 0  # More input than ET

        # Storage should increase (or stay similar if capacity limited)
        assert storage_final >= storage_initial * 0.95

    def test_mixed_connectivity(self):
        """Test water balance with mixed connectivity."""
        n_layers = 4
        theta = np.full(n_layers, 0.25)
        thickness_arr = np.full(n_layers, 0.25)
        dt = 3600

        # Alternating connectivity
        kappa = np.array([1.0, 0.1, 0.9, 0.2])

        flux_in = np.array([1e-5, 5e-6, 5e-6, 2e-6])
        flux_out = np.array([5e-6, 2e-6, 2e-6, 1e-6])
        ET = np.full(n_layers, 1e-6)

        theta_new = calculate_storage_change(
            theta, flux_in, flux_out, ET, thickness_arr, dt, kappa
        )

        # Connected layers should respond to fluxes
        # Disconnected layers should only respond to ET

        # Layer 0 (connected): should increase
        assert theta_new[0] > theta[0]

        # Layer 1 (disconnected): should decrease (ET only)
        assert theta_new[1] < theta[1]

        # Layer 2 (connected): should increase
        assert theta_new[2] > theta[2]

        # Layer 3 (disconnected): should decrease (ET only)
        assert theta_new[3] < theta[3]

    def test_water_partitioning(self):
        """Test partitioning between mobile and trapped water."""
        n_layers = 6
        theta = np.array([0.30, 0.25, 0.20, 0.15, 0.35, 0.28])
        kappa = np.array([0.9, 0.3, 0.8, 0.2, 0.95, 0.1])
        thickness_arr = np.full(n_layers, 0.2)

        mobile = calculate_mobile_water(theta, kappa, thickness_arr)
        trapped = calculate_trapped_water(theta, kappa, thickness_arr)
        total = np.sum(theta * thickness_arr)

        # Mobile + trapped should equal total
        assert mobile + trapped == pytest.approx(total)

        # Should have both mobile and trapped water
        assert mobile > 0
        assert trapped > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
