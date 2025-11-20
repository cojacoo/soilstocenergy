"""
Unit tests for connectivity module.

Tests cover:
- E_crit calculation
- Connectivity state calculation (binary, sigmoid, linear)
- Bond activation
- ConnectivityState class
- Hysteresis
"""

import numpy as np
import pytest
from soilstocenergy.core.connectivity import (
    ConnectivityCalculator,
    ConnectivityState,
    HysteresisConnectivity,
)


class TestConnectivityCalculator:
    """Test connectivity calculator."""

    def test_initialization(self):
        """Test initialization with default parameters."""
        calc = ConnectivityCalculator()
        assert calc.beta > 0
        assert calc.default_mode in ['binary', 'sigmoid', 'linear']

    def test_E_crit_base_case(self):
        """Test E_crit calculation with no structure."""
        calc = ConnectivityCalculator()

        E_crit = calc.calculate_E_crit(
            macroporosity=0.0,
            rDUNE=0.0,
            E_base=-1000.0
        )

        assert E_crit == pytest.approx(-1000.0)

    def test_E_crit_with_macroporosity(self):
        """Test that macroporosity lowers E_crit."""
        calc = ConnectivityCalculator()

        E_crit_no_macro = calc.calculate_E_crit(
            macroporosity=0.0,
            rDUNE=0.0,
            E_base=-1000.0,
            alpha_macro=500.0
        )

        E_crit_with_macro = calc.calculate_E_crit(
            macroporosity=0.1,
            rDUNE=0.0,
            E_base=-1000.0,
            alpha_macro=500.0
        )

        # Higher macroporosity → lower E_crit (easier activation)
        assert E_crit_with_macro < E_crit_no_macro

    def test_E_crit_with_rDUNE(self):
        """Test that higher rDUNE lowers E_crit."""
        calc = ConnectivityCalculator()

        E_crit_low_rDUNE = calc.calculate_E_crit(
            macroporosity=0.0,
            rDUNE=0.5,
            E_base=-1000.0,
            beta_rDUNE=100.0
        )

        E_crit_high_rDUNE = calc.calculate_E_crit(
            macroporosity=0.0,
            rDUNE=2.0,
            E_base=-1000.0,
            beta_rDUNE=100.0
        )

        # Higher rDUNE → lower E_crit (favorable drainage)
        assert E_crit_high_rDUNE < E_crit_low_rDUNE

    def test_binary_connectivity(self):
        """Test binary connectivity function."""
        calc = ConnectivityCalculator()

        # Above threshold
        kappa = calc.calculate_connectivity(
            E_free=0.0,
            E_crit=-100.0,
            mode='binary'
        )
        assert kappa == pytest.approx(1.0)

        # Below threshold
        kappa = calc.calculate_connectivity(
            E_free=-200.0,
            E_crit=-100.0,
            mode='binary'
        )
        assert kappa == pytest.approx(0.0)

        # At threshold (should be disconnected)
        kappa = calc.calculate_connectivity(
            E_free=-100.0,
            E_crit=-100.0,
            mode='binary'
        )
        assert kappa == pytest.approx(0.0)

    def test_sigmoid_connectivity(self):
        """Test sigmoid connectivity function."""
        calc = ConnectivityCalculator(transition_sharpness=10.0)

        # Well above threshold
        kappa = calc.calculate_connectivity(
            E_free=100.0,
            E_crit=-100.0,
            mode='sigmoid'
        )
        assert kappa > 0.99

        # Well below threshold
        kappa = calc.calculate_connectivity(
            E_free=-300.0,
            E_crit=-100.0,
            mode='sigmoid'
        )
        assert kappa < 0.01

        # At threshold (should be 0.5)
        kappa = calc.calculate_connectivity(
            E_free=-100.0,
            E_crit=-100.0,
            mode='sigmoid'
        )
        assert kappa == pytest.approx(0.5, rel=0.01)

    def test_sigmoid_is_continuous(self):
        """Test that sigmoid provides continuous transition."""
        calc = ConnectivityCalculator(transition_sharpness=10.0)

        E_crit = -100.0
        E_range = np.linspace(-200.0, 0.0, 100)

        kappa = calc.calculate_connectivity(E_range, E_crit, mode='sigmoid')

        # Should be monotonically increasing
        assert np.all(np.diff(kappa) >= 0)

        # Should span [0, 1]
        assert kappa[0] < 0.1
        assert kappa[-1] > 0.9

    def test_linear_connectivity(self):
        """Test linear connectivity function."""
        calc = ConnectivityCalculator()

        E_crit = -100.0

        # Well above
        kappa = calc.calculate_connectivity(
            E_free=0.0,
            E_crit=E_crit,
            mode='linear'
        )
        assert kappa == pytest.approx(1.0)

        # Well below
        kappa = calc.calculate_connectivity(
            E_free=-300.0,
            E_crit=E_crit,
            mode='linear'
        )
        assert kappa == pytest.approx(0.0)

    def test_connectivity_with_arrays(self):
        """Test connectivity calculation with array inputs."""
        calc = ConnectivityCalculator()

        E_free = np.array([-200.0, -100.0, 0.0, 100.0])
        E_crit = -100.0

        kappa = calc.calculate_connectivity(E_free, E_crit, mode='binary')

        assert kappa.shape == E_free.shape
        expected = np.array([0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(kappa, expected)

    def test_bond_activation(self):
        """Test bond activation between two elements."""
        calc = ConnectivityCalculator()

        # Both fully connected
        p_bond = calc.calculate_bond_activation(kappa_i=1.0, kappa_j=1.0)
        assert p_bond == pytest.approx(1.0)

        # One disconnected
        p_bond = calc.calculate_bond_activation(kappa_i=1.0, kappa_j=0.0)
        assert p_bond == pytest.approx(0.0)

        # Both partially connected
        p_bond = calc.calculate_bond_activation(kappa_i=0.5, kappa_j=0.5)
        assert p_bond == pytest.approx(0.25)

    def test_bond_activation_with_barrier(self):
        """Test bond activation with energy barrier."""
        calc = ConnectivityCalculator()

        # Above barrier
        p_bond = calc.calculate_bond_activation(
            kappa_i=1.0,
            kappa_j=1.0,
            E_barrier=0.0,
            E_interface=10.0
        )
        assert p_bond == pytest.approx(1.0)

        # Below barrier
        p_bond = calc.calculate_bond_activation(
            kappa_i=1.0,
            kappa_j=1.0,
            E_barrier=0.0,
            E_interface=-10.0
        )
        assert p_bond == pytest.approx(0.0)


class TestConnectivityState:
    """Test ConnectivityState class."""

    def test_initialization(self):
        """Test initialization."""
        state = ConnectivityState(shape=(10,))

        assert state.shape == (10,)
        assert state.kappa.shape == (10,)
        assert state.E_crit.shape == (10,)

    def test_set_E_crit_uniform(self):
        """Test setting uniform E_crit."""
        state = ConnectivityState(shape=(10,))
        state.set_E_crit(E_crit=-100.0)

        assert np.all(state.E_crit == -100.0)

    def test_set_E_crit_array(self):
        """Test setting spatially variable E_crit."""
        state = ConnectivityState(shape=(10,))
        E_crit_array = np.linspace(-200, -50, 10)
        state.set_E_crit(E_crit=E_crit_array)

        np.testing.assert_array_almost_equal(state.E_crit, E_crit_array)

    def test_set_E_crit_from_structure(self):
        """Test calculating E_crit from macroporosity and rDUNE."""
        state = ConnectivityState(shape=(10,))

        macroporosity = np.linspace(0.0, 0.2, 10)
        rDUNE = np.linspace(0.0, 2.0, 10)

        state.set_E_crit(
            E_crit=None,
            macroporosity=macroporosity,
            rDUNE=rDUNE,
            E_base=-1000.0
        )

        # E_crit should decrease with macroporosity and rDUNE
        assert state.E_crit[0] > state.E_crit[-1]

    def test_update_connectivity(self):
        """Test updating connectivity based on E_free."""
        state = ConnectivityState(shape=(5,))
        state.set_E_crit(E_crit=-100.0)

        # Set free energy field
        E_free = np.array([-200, -150, -100, -50, 0])
        state.update(E_free, mode='binary')

        # First 3 should be disconnected, last 2 connected
        expected = np.array([0, 0, 0, 1, 1])
        np.testing.assert_array_almost_equal(state.kappa, expected)

    def test_get_active_fraction(self):
        """Test calculation of active fraction."""
        state = ConnectivityState(shape=(10,))
        state.kappa = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        fraction = state.get_active_fraction(threshold=0.5)
        assert fraction == pytest.approx(0.5)

    def test_get_active_mask(self):
        """Test active mask generation."""
        state = ConnectivityState(shape=(5,))
        state.kappa = np.array([0.0, 0.3, 0.5, 0.7, 1.0])

        mask = state.get_active_mask(threshold=0.5)

        # Only last two should be active
        expected = np.array([False, False, False, True, True])
        np.testing.assert_array_equal(mask, expected)

    def test_get_connectivity_pattern(self):
        """Test connectivity pattern statistics."""
        state = ConnectivityState(shape=(10,))
        state.kappa = np.linspace(0, 1, 10)

        pattern = state.get_connectivity_pattern()

        assert 'mean' in pattern
        assert 'std' in pattern
        assert 'active_fraction' in pattern
        assert 0 <= pattern['mean'] <= 1
        assert 0 <= pattern['active_fraction'] <= 1

    def test_percolation_probability(self):
        """Test percolation probability estimation."""
        state = ConnectivityState(shape=(10,))

        # Low connectivity → low percolation probability
        state.kappa = np.full(10, 0.1)
        p_perc_low = state.get_percolation_probability()

        # High connectivity → high percolation probability
        state.kappa = np.full(10, 0.9)
        p_perc_high = state.get_percolation_probability()

        assert p_perc_high > p_perc_low

    def test_2d_connectivity_state(self):
        """Test with 2D domain."""
        state = ConnectivityState(shape=(5, 5))

        assert state.shape == (5, 5)
        assert state.kappa.shape == (5, 5)

        # Set uniform E_crit
        state.set_E_crit(E_crit=-100.0)

        # Update with 2D field
        E_free = np.random.randn(5, 5) * 50
        state.update(E_free, mode='sigmoid')

        assert state.kappa.shape == (5, 5)


class TestHysteresisConnectivity:
    """Test hysteresis connectivity."""

    def test_initialization(self):
        """Test initialization with hysteresis."""
        hyst = HysteresisConnectivity(
            E_crit_wet=-50.0,
            E_crit_dry=-150.0
        )

        assert hyst.E_crit_wet == -50.0
        assert hyst.E_crit_dry == -150.0

    def test_invalid_hysteresis(self):
        """Test that initialization fails if E_crit_dry >= E_crit_wet."""
        with pytest.raises(ValueError):
            HysteresisConnectivity(
                E_crit_wet=-150.0,
                E_crit_dry=-50.0  # Invalid: dry > wet
            )

    def test_wetting_uses_higher_threshold(self):
        """Test that wetting uses E_crit_wet."""
        hyst = HysteresisConnectivity(
            E_crit_wet=-50.0,
            E_crit_dry=-150.0
        )

        # Simulate wetting: increasing E_free
        hyst.previous_E_free = -200.0
        kappa = hyst.calculate_connectivity(E_free=-100.0, mode='binary')

        # -100 is above E_crit_dry but below E_crit_wet
        # During wetting, should use E_crit_wet, so disconnected
        assert kappa == pytest.approx(0.0)

    def test_drying_uses_lower_threshold(self):
        """Test that drying uses E_crit_dry."""
        hyst = HysteresisConnectivity(
            E_crit_wet=-50.0,
            E_crit_dry=-150.0
        )

        # Simulate drying: decreasing E_free
        hyst.previous_E_free = 0.0
        kappa = hyst.calculate_connectivity(E_free=-100.0, mode='binary')

        # -100 is above E_crit_dry but below E_crit_wet
        # During drying, should use E_crit_dry, so connected
        assert kappa == pytest.approx(1.0)

    def test_hysteresis_width(self):
        """Test hysteresis width calculation."""
        hyst = HysteresisConnectivity(
            E_crit_wet=-50.0,
            E_crit_dry=-150.0
        )

        width = hyst.get_hysteresis_width()
        assert width == pytest.approx(100.0)

    def test_hysteresis_with_arrays(self):
        """Test hysteresis with spatial arrays."""
        E_crit_wet = np.array([-50.0, -60.0, -70.0])
        E_crit_dry = np.array([-150.0, -160.0, -170.0])

        hyst = HysteresisConnectivity(
            E_crit_wet=E_crit_wet,
            E_crit_dry=E_crit_dry
        )

        width = hyst.get_hysteresis_width()
        expected_width = np.array([100.0, 100.0, 100.0])
        np.testing.assert_array_almost_equal(width, expected_width)


class TestConnectivityComparison:
    """Compare different connectivity formulations."""

    def test_binary_vs_sigmoid_at_threshold(self):
        """Test that binary is 0 at threshold while sigmoid is 0.5."""
        calc = ConnectivityCalculator()

        E_crit = -100.0
        E_free = -100.0

        kappa_binary = calc.calculate_connectivity(E_free, E_crit, mode='binary')
        kappa_sigmoid = calc.calculate_connectivity(E_free, E_crit, mode='sigmoid')

        assert kappa_binary == pytest.approx(0.0)
        assert kappa_sigmoid == pytest.approx(0.5, rel=0.01)

    def test_all_modes_agree_far_from_threshold(self):
        """Test that all modes agree far from threshold."""
        calc = ConnectivityCalculator()

        E_crit = -100.0

        # Well above threshold
        E_free = 100.0
        kappa_binary = calc.calculate_connectivity(E_free, E_crit, mode='binary')
        kappa_sigmoid = calc.calculate_connectivity(E_free, E_crit, mode='sigmoid')
        kappa_linear = calc.calculate_connectivity(E_free, E_crit, mode='linear')

        assert kappa_binary == pytest.approx(1.0)
        assert kappa_sigmoid == pytest.approx(1.0, abs=0.01)
        assert kappa_linear == pytest.approx(1.0)

        # Well below threshold
        E_free = -300.0
        kappa_binary = calc.calculate_connectivity(E_free, E_crit, mode='binary')
        kappa_sigmoid = calc.calculate_connectivity(E_free, E_crit, mode='sigmoid')
        kappa_linear = calc.calculate_connectivity(E_free, E_crit, mode='linear')

        assert kappa_binary == pytest.approx(0.0)
        assert kappa_sigmoid == pytest.approx(0.0, abs=0.01)
        assert kappa_linear == pytest.approx(0.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
