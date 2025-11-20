"""
Unit tests for thermodynamics module.

Tests cover:
- Van Genuchten retention curve
- Brooks-Corey retention curve
- Free energy calculations
- Local equilibrium
- rDUNE index
"""

import numpy as np
import pytest
from soilstocenergy.core.thermodynamics import (
    VanGenuchten,
    BrooksCorey,
    FreeEnergyCalculator,
    LocalEquilibrium,
    calculate_rDUNE,
    calculate_HAND_from_elevation,
)


class TestVanGenuchten:
    """Test van Genuchten retention curve."""

    def test_initialization(self):
        """Test initialization with valid parameters."""
        vg = VanGenuchten(theta_r=0.05, theta_s=0.45, alpha=2.0, n=1.5)
        assert vg.theta_r == 0.05
        assert vg.theta_s == 0.45
        assert vg.alpha == 2.0
        assert vg.n == 1.5
        assert vg.m == pytest.approx(1.0 - 1.0/1.5)

    def test_saturated_conditions(self):
        """Test that θ = θ_s when ψ >= 0."""
        vg = VanGenuchten(theta_r=0.05, theta_s=0.45, alpha=2.0, n=1.5)

        # At saturation
        theta = vg.psi_to_theta(0.0)
        assert theta == pytest.approx(0.45)

        # Above saturation (ponding)
        theta = vg.psi_to_theta(0.5)
        assert theta == pytest.approx(0.45)

    def test_theta_to_psi_to_theta(self):
        """Test round-trip conversion θ → ψ → θ."""
        vg = VanGenuchten(theta_r=0.05, theta_s=0.45, alpha=2.0, n=1.5)

        theta_original = 0.30
        psi = vg.theta_to_psi(theta_original)
        theta_recovered = vg.psi_to_theta(psi)

        assert theta_recovered == pytest.approx(theta_original, rel=1e-6)

    def test_array_input(self):
        """Test that arrays are handled correctly."""
        vg = VanGenuchten(theta_r=0.05, theta_s=0.45, alpha=2.0, n=1.5)

        theta_array = np.array([0.10, 0.20, 0.30, 0.40])
        psi_array = vg.theta_to_psi(theta_array)

        assert psi_array.shape == theta_array.shape
        assert np.all(psi_array < 0)  # All unsaturated

    def test_effective_saturation(self):
        """Test effective saturation calculation."""
        vg = VanGenuchten(theta_r=0.05, theta_s=0.45, alpha=2.0, n=1.5)

        # At residual
        Se = vg.effective_saturation(0.05)
        assert Se == pytest.approx(0.0)

        # At saturation
        Se = vg.effective_saturation(0.45)
        assert Se == pytest.approx(1.0)

        # Midpoint
        Se = vg.effective_saturation(0.25)
        assert 0 < Se < 1


class TestBrooksCorey:
    """Test Brooks-Corey retention curve."""

    def test_initialization(self):
        """Test initialization with valid parameters."""
        bc = BrooksCorey(theta_r=0.05, theta_s=0.45, psi_b=0.2, lambda_=2.0)
        assert bc.theta_r == 0.05
        assert bc.theta_s == 0.45
        assert bc.psi_b == 0.2
        assert bc.lambda_ == 2.0

    def test_saturated_conditions(self):
        """Test saturated conditions below air entry pressure."""
        bc = BrooksCorey(theta_r=0.05, theta_s=0.45, psi_b=0.2, lambda_=2.0)

        # Below air entry pressure
        theta = bc.psi_to_theta(-0.1)
        assert theta == pytest.approx(0.45)

        # At air entry
        theta = bc.psi_to_theta(-0.2)
        assert theta == pytest.approx(0.45)

    def test_unsaturated_conditions(self):
        """Test unsaturated conditions above air entry pressure."""
        bc = BrooksCorey(theta_r=0.05, theta_s=0.45, psi_b=0.2, lambda_=2.0)

        # Above air entry pressure
        theta = bc.psi_to_theta(-0.5)
        assert bc.theta_r < theta < bc.theta_s

    def test_theta_to_psi_to_theta(self):
        """Test round-trip conversion."""
        bc = BrooksCorey(theta_r=0.05, theta_s=0.45, psi_b=0.2, lambda_=2.0)

        theta_original = 0.30
        psi = bc.theta_to_psi(theta_original)
        theta_recovered = bc.psi_to_theta(psi)

        assert theta_recovered == pytest.approx(theta_original, rel=1e-6)


class TestFreeEnergyCalculator:
    """Test free energy calculator."""

    def test_initialization_vangenuchten(self):
        """Test initialization with van Genuchten model."""
        fe_calc = FreeEnergyCalculator(
            theta_r=0.05, theta_s=0.45, alpha=2.0, n=1.5,
            model='vanGenuchten'
        )
        assert fe_calc.model == 'vanGenuchten'
        assert isinstance(fe_calc.retention_curve, VanGenuchten)

    def test_initialization_brookscorey(self):
        """Test initialization with Brooks-Corey model."""
        fe_calc = FreeEnergyCalculator(
            theta_r=0.05, theta_s=0.45, psi_b=0.2, lambda_=2.0,
            model='brooksCorey'
        )
        assert fe_calc.model == 'brooksCorey'
        assert isinstance(fe_calc.retention_curve, BrooksCorey)

    def test_free_energy_calculation(self):
        """Test free energy calculation."""
        fe_calc = FreeEnergyCalculator(
            theta_r=0.05, theta_s=0.45, alpha=2.0, n=1.5,
            model='vanGenuchten'
        )

        # Calculate free energy
        theta = 0.30
        HAND = 2.0
        E_free = fe_calc.calculate_free_energy(theta, HAND)

        # E_free should be in J/m³ (Pa)
        # With HAND > 0, gravitational component adds positive energy
        assert isinstance(E_free, (float, np.floating))

    def test_free_energy_increases_with_HAND(self):
        """Test that free energy increases with HAND (gravitational potential)."""
        fe_calc = FreeEnergyCalculator(
            theta_r=0.05, theta_s=0.45, alpha=2.0, n=1.5,
            model='vanGenuchten'
        )

        theta = 0.30
        E1 = fe_calc.calculate_free_energy(theta, HAND=0.0)
        E2 = fe_calc.calculate_free_energy(theta, HAND=1.0)
        E3 = fe_calc.calculate_free_energy(theta, HAND=2.0)

        # Higher position → higher free energy
        assert E1 < E2 < E3

    def test_free_energy_increases_with_theta(self):
        """Test that free energy increases with water content."""
        fe_calc = FreeEnergyCalculator(
            theta_r=0.05, theta_s=0.45, alpha=2.0, n=1.5,
            model='vanGenuchten'
        )

        HAND = 1.0
        E1 = fe_calc.calculate_free_energy(0.10, HAND)
        E2 = fe_calc.calculate_free_energy(0.25, HAND)
        E3 = fe_calc.calculate_free_energy(0.40, HAND)

        # Wetter soil → less negative matric potential → higher free energy
        assert E1 < E2 < E3

    def test_storage_state(self):
        """Test storage state classification."""
        fe_calc = FreeEnergyCalculator(
            theta_r=0.05, theta_s=0.45, alpha=2.0, n=1.5,
            model='vanGenuchten'
        )

        E_eq = 0.0

        # Excess
        state = fe_calc.storage_state(E_free=100.0, E_equilibrium=E_eq)
        assert state == 'excess'

        # Deficit
        state = fe_calc.storage_state(E_free=-100.0, E_equilibrium=E_eq)
        assert state == 'deficit'

        # Equilibrium
        state = fe_calc.storage_state(E_free=0.5, E_equilibrium=E_eq)
        assert state == 'equilibrium'

    def test_array_input(self):
        """Test with array inputs."""
        fe_calc = FreeEnergyCalculator(
            theta_r=0.05, theta_s=0.45, alpha=2.0, n=1.5,
            model='vanGenuchten'
        )

        theta_array = np.array([0.10, 0.20, 0.30, 0.40])
        HAND_array = np.array([0.0, 1.0, 2.0, 3.0])

        E_free = fe_calc.calculate_free_energy(theta_array, HAND_array)

        assert E_free.shape == theta_array.shape
        assert np.all(np.isfinite(E_free))


class TestLocalEquilibrium:
    """Test local equilibrium calculator."""

    def test_initialization(self):
        """Test initialization."""
        fe_calc = FreeEnergyCalculator(
            theta_r=0.05, theta_s=0.45, alpha=2.0, n=1.5,
            model='vanGenuchten'
        )
        le = LocalEquilibrium(fe_calc, reference_theta=0.30, reference_HAND=0.0)

        assert le.reference_theta == 0.30
        assert le.reference_HAND == 0.0
        assert hasattr(le, 'E_reference')

    def test_equilibrium_increases_with_HAND(self):
        """Test that equilibrium energy increases with HAND."""
        fe_calc = FreeEnergyCalculator(
            theta_r=0.05, theta_s=0.45, alpha=2.0, n=1.5,
            model='vanGenuchten'
        )
        le = LocalEquilibrium(fe_calc)

        E_eq_0 = le.equilibrium_free_energy(HAND=0.0)
        E_eq_1 = le.equilibrium_free_energy(HAND=1.0)
        E_eq_2 = le.equilibrium_free_energy(HAND=2.0)

        assert E_eq_0 < E_eq_1 < E_eq_2

    def test_storage_excess_deficit(self):
        """Test storage excess/deficit classification."""
        fe_calc = FreeEnergyCalculator(
            theta_r=0.05, theta_s=0.45, alpha=2.0, n=1.5,
            model='vanGenuchten'
        )
        le = LocalEquilibrium(fe_calc, reference_theta=0.30, reference_HAND=0.0)

        # Very wet soil → storage excess
        is_excess = le.is_storage_excess(theta=0.40, HAND=0.0)
        assert is_excess

        # Very dry soil → storage deficit
        is_deficit = le.is_storage_deficit(theta=0.10, HAND=0.0)
        assert is_deficit


class TestRDUNE:
    """Test rDUNE index calculation."""

    def test_rdune_calculation(self):
        """Test rDUNE calculation."""
        HAND = 2.0
        flow_path_length = 10.0

        rDUNE = calculate_rDUNE(HAND, flow_path_length)

        # rDUNE = -ln(2.0 / 10.0) = -ln(0.2) ≈ 1.609
        assert rDUNE == pytest.approx(-np.log(0.2), rel=1e-6)

    def test_rdune_array_input(self):
        """Test rDUNE with array inputs."""
        HAND = np.array([1.0, 2.0, 3.0])
        flow_path_length = np.array([10.0, 20.0, 30.0])

        rDUNE = calculate_rDUNE(HAND, flow_path_length)

        assert rDUNE.shape == HAND.shape
        assert np.all(rDUNE > 0)  # All should be positive

    def test_rdune_increases_with_dissipation(self):
        """Test that rDUNE increases with longer flow path (more dissipation)."""
        HAND = 2.0

        rDUNE_short = calculate_rDUNE(HAND, flow_path_length=5.0)
        rDUNE_long = calculate_rDUNE(HAND, flow_path_length=20.0)

        # Longer path → more dissipation → higher rDUNE
        assert rDUNE_long > rDUNE_short


class TestHAND:
    """Test HAND calculation."""

    def test_hand_calculation(self):
        """Test HAND calculation from elevation."""
        elevation = np.array([100.0, 105.0, 110.0, 115.0])
        stream_elevation = 100.0

        HAND = calculate_HAND_from_elevation(elevation, stream_elevation)

        expected = np.array([0.0, 5.0, 10.0, 15.0])
        np.testing.assert_array_almost_equal(HAND, expected)

    def test_hand_with_variable_stream(self):
        """Test HAND with spatially variable stream elevation."""
        elevation = np.array([100.0, 105.0, 110.0, 115.0])
        stream_elevation = np.array([100.0, 102.0, 105.0, 108.0])

        HAND = calculate_HAND_from_elevation(elevation, stream_elevation)

        expected = np.array([0.0, 3.0, 5.0, 7.0])
        np.testing.assert_array_almost_equal(HAND, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
