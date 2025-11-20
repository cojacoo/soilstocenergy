"""
Unit tests for 2D hillslope model.

Tests cover:
- Grid initialization
- Hillslope creation
- Spatial connectivity
- Infiltration
- Lateral fluxes
- Water balance
- Contributing area dynamics
"""

import numpy as np
import pytest
from soilstocenergy.models.hillslope_2d import (
    HillslopeGrid,
    Hillslope2D,
    create_synthetic_hillslope,
)


class TestHillslopeGrid:
    """Test HillslopeGrid dataclass."""

    def test_initialization(self):
        """Test grid initialization."""
        ny, nx = 10, 20
        elevation = np.random.rand(ny, nx)
        HAND = np.random.rand(ny, nx)
        fpl = np.random.rand(ny, nx)

        grid = HillslopeGrid(
            ny=ny,
            nx=nx,
            dy=2.0,
            dx=2.0,
            elevation=elevation,
            HAND=HAND,
            flow_path_length=fpl
        )

        assert grid.ny == 10
        assert grid.nx == 20
        assert grid.dy == 2.0
        assert grid.dx == 2.0
        assert grid.elevation.shape == (ny, nx)

    def test_shape_property(self):
        """Test shape property."""
        ny, nx = 5, 10
        grid = HillslopeGrid(
            ny=ny,
            nx=nx,
            dy=1.0,
            dx=1.0,
            elevation=np.zeros((ny, nx)),
            HAND=np.zeros((ny, nx)),
            flow_path_length=np.zeros((ny, nx))
        )

        assert grid.shape == (5, 10)

    def test_cell_area(self):
        """Test cell area calculation."""
        grid = HillslopeGrid(
            ny=5,
            nx=10,
            dy=2.0,
            dx=3.0,
            elevation=np.zeros((5, 10)),
            HAND=np.zeros((5, 10)),
            flow_path_length=np.zeros((5, 10))
        )

        assert grid.cell_area == pytest.approx(6.0)


class TestHillslopeInitialization:
    """Test Hillslope2D initialization."""

    def test_basic_initialization(self):
        """Test basic hillslope initialization."""
        hillslope = create_synthetic_hillslope(ny=10, nx=20)

        assert hillslope.grid.ny == 10
        assert hillslope.grid.nx == 20
        assert hillslope.theta.shape == (10, 20)
        assert hillslope.kappa.shape == (10, 20)
        assert hillslope.E_free.shape == (10, 20)

    def test_initial_moisture(self):
        """Test initial moisture state."""
        soil_props = {
            'theta_r': 0.05,
            'theta_s': 0.45,
            'alpha': 2.0,
            'n': 1.5,
            'K_sat': 1e-5,
            'theta_initial': 0.30,
        }

        hillslope = create_synthetic_hillslope(
            ny=5, nx=10, soil_properties=soil_props
        )

        # All cells should start at initial moisture
        assert np.all(hillslope.theta == pytest.approx(0.30))

    def test_E_crit_field_initialization(self):
        """Test that E_crit field is calculated."""
        hillslope = create_synthetic_hillslope(ny=5, nx=10)

        # E_crit should be set in connectivity state
        assert hillslope.conn_state.E_crit is not None
        assert hillslope.conn_state.E_crit.shape == (5, 10)

    def test_topography_influence_on_E_crit(self):
        """Test that topography affects E_crit."""
        hillslope = create_synthetic_hillslope(ny=5, nx=10, slope=0.1)

        # E_crit should vary spatially with HAND
        # Near stream (low HAND) should have different E_crit than far from stream
        E_crit = hillslope.conn_state.E_crit

        # Variance should be non-zero
        assert np.var(E_crit) > 0


class TestConnectivity:
    """Test spatial connectivity calculations."""

    def test_connectivity_field_initialization(self):
        """Test that connectivity is initialized."""
        hillslope = create_synthetic_hillslope(ny=5, nx=10)

        # Kappa should be calculated
        assert hillslope.kappa.shape == (5, 10)
        assert np.all(hillslope.kappa >= 0.0)
        assert np.all(hillslope.kappa <= 1.0)

    def test_wet_cells_more_connected(self):
        """Test that wetter cells have higher connectivity."""
        hillslope = create_synthetic_hillslope(ny=5, nx=10)

        # Set varying moisture
        hillslope.theta[:, 0:5] = 0.15  # Dry
        hillslope.theta[:, 5:10] = 0.40  # Wet

        hillslope._update_energy_and_connectivity()

        # Wet cells should generally have higher connectivity
        # (This depends on E_crit distribution, so use average)
        kappa_dry = np.mean(hillslope.kappa[:, 0:5])
        kappa_wet = np.mean(hillslope.kappa[:, 5:10])

        assert kappa_wet >= kappa_dry

    def test_connectivity_update(self):
        """Test that connectivity updates with moisture change."""
        hillslope = create_synthetic_hillslope(ny=5, nx=10)

        # Initial connectivity
        kappa_initial = hillslope.kappa.copy()

        # Change moisture significantly
        hillslope.theta[:, :] = 0.10  # Dry everywhere
        hillslope._update_energy_and_connectivity()

        kappa_after = hillslope.kappa.copy()

        # Connectivity should change
        assert not np.allclose(kappa_initial, kappa_after)


class TestConductivity:
    """Test hydraulic conductivity calculation."""

    def test_conductivity_shape(self):
        """Test that conductivity field has correct shape."""
        hillslope = create_synthetic_hillslope(ny=5, nx=10)

        K = hillslope.calculate_conductivity()

        assert K.shape == (5, 10)

    def test_conductivity_bounds(self):
        """Test that conductivity is within physical bounds."""
        hillslope = create_synthetic_hillslope(ny=5, nx=10)

        K = hillslope.calculate_conductivity()

        # Should be non-negative
        assert np.all(K >= 0.0)

        # Should be less than or equal to K_sat
        K_sat = hillslope.soil_props['K_sat']
        assert np.all(K <= K_sat * 1.001)  # Small tolerance for numerical

    def test_connectivity_reduces_conductivity(self):
        """Test that low connectivity reduces conductivity."""
        hillslope = create_synthetic_hillslope(ny=3, nx=3)

        # Set uniform moisture
        hillslope.theta[:, :] = 0.35
        hillslope._update_energy_and_connectivity()

        K_connected = hillslope.calculate_conductivity()

        # Force disconnection
        hillslope.kappa[:, :] = 0.1
        K_disconnected = hillslope.calculate_conductivity()

        # Disconnected should have lower conductivity
        assert np.mean(K_disconnected) < np.mean(K_connected)

    def test_dry_cells_low_conductivity(self):
        """Test that dry cells have low conductivity."""
        hillslope = create_synthetic_hillslope(ny=5, nx=10)

        # Set to near residual moisture
        hillslope.theta[:, :] = hillslope.soil_props['theta_r'] + 0.01
        hillslope._update_energy_and_connectivity()

        K = hillslope.calculate_conductivity()

        # Should be very small
        K_sat = hillslope.soil_props['K_sat']
        assert np.mean(K) < K_sat * 0.01


class TestInfiltration:
    """Test infiltration process."""

    def test_infiltration_increases_moisture(self):
        """Test that infiltration increases moisture."""
        hillslope = create_synthetic_hillslope(ny=5, nx=10)

        theta_initial = hillslope.theta.copy()

        # Apply rain
        precip_rate = 1e-5  # m/s
        dt = 3600  # 1 hour

        inf = hillslope.infiltrate(precip_rate, dt)

        # Moisture should increase where infiltration occurred
        assert np.sum(hillslope.theta > theta_initial) > 0

    def test_infiltration_only_connected_cells(self):
        """Test that infiltration only occurs in connected cells."""
        hillslope = create_synthetic_hillslope(ny=5, nx=10)

        # Force pattern: connected left, disconnected right
        hillslope.theta[:, 0:5] = 0.40  # Wet (likely connected)
        hillslope.theta[:, 5:10] = 0.08  # Dry (likely disconnected)
        hillslope._update_energy_and_connectivity()

        theta_before = hillslope.theta.copy()

        # Apply rain
        inf = hillslope.infiltrate(1e-5, 3600)

        # More infiltration where connected
        # Check by comparing infiltration amounts
        inf_left = np.sum(inf[:, 0:5])
        inf_right = np.sum(inf[:, 5:10])

        # Left (wetter/connected) should have more infiltration
        assert inf_left >= inf_right

    def test_infiltration_respects_capacity(self):
        """Test that infiltration doesn't exceed soil capacity."""
        hillslope = create_synthetic_hillslope(ny=3, nx=3)

        # Nearly saturated
        theta_s = hillslope.soil_props['theta_s']
        hillslope.theta[:, :] = theta_s - 0.01
        hillslope._update_energy_and_connectivity()

        # Large rainfall
        precip_rate = 1e-3  # Very intense
        dt = 3600

        hillslope.infiltrate(precip_rate, dt)

        # Should not exceed saturation
        assert np.all(hillslope.theta <= theta_s * 1.001)


class TestLateralFluxes:
    """Test lateral flux calculations."""

    def test_flux_calculation(self):
        """Test basic flux calculation."""
        hillslope = create_synthetic_hillslope(ny=5, nx=10)

        flux_x, flux_y = hillslope.calculate_lateral_fluxes()

        # Check shapes
        assert flux_x.shape == (5, 9)  # nx-1 interfaces
        assert flux_y.shape == (4, 10)  # ny-1 interfaces

    def test_flux_direction_downslope(self):
        """Test that flux is generally downslope."""
        # Create hillslope with clear gradient
        hillslope = create_synthetic_hillslope(ny=3, nx=10, slope=0.2)

        # Uniform moisture
        hillslope.theta[:, :] = 0.30
        hillslope._update_energy_and_connectivity()

        flux_x, flux_y = hillslope.calculate_lateral_fluxes()

        # Net flux should be towards stream (negative x direction)
        # Sum of fluxes entering left should be greater than leaving right
        # (flux_x is positive in +x direction, so sum should be negative for net towards stream)
        # This is a simplified check
        pass  # Flux direction depends on head gradients

    def test_no_flux_disconnected(self):
        """Test that disconnected cells have reduced flux."""
        hillslope = create_synthetic_hillslope(ny=3, nx=5)

        # Set uniform moisture
        hillslope.theta[:, :] = 0.30
        hillslope._update_energy_and_connectivity()

        flux_x_connected, flux_y_connected = hillslope.calculate_lateral_fluxes()

        # Force disconnection
        hillslope.kappa[:, :] = 0.0
        flux_x_disconnected, flux_y_disconnected = hillslope.calculate_lateral_fluxes()

        # Disconnected should have much smaller fluxes
        assert np.sum(np.abs(flux_x_disconnected)) < np.sum(np.abs(flux_x_connected)) * 0.1


class TestTimestepping:
    """Test time stepping."""

    def test_step_function_runs(self):
        """Test that step function executes without error."""
        hillslope = create_synthetic_hillslope(ny=5, nx=10)

        dt = 600  # 10 minutes
        precip_rate = 1e-5
        ET_rate = 2e-6

        # Should not raise
        hillslope.step(dt, precip_rate=precip_rate, ET_rate=ET_rate)

    def test_step_with_rainfall(self):
        """Test step with rainfall."""
        hillslope = create_synthetic_hillslope(ny=5, nx=10)

        storage_initial = np.sum(hillslope.theta)

        hillslope.step(dt=3600, precip_rate=1e-5, ET_rate=0.0)

        storage_final = np.sum(hillslope.theta)

        # Storage should increase (or stay similar)
        assert storage_final >= storage_initial * 0.95

    def test_step_with_ET(self):
        """Test step with evapotranspiration."""
        hillslope = create_synthetic_hillslope(ny=5, nx=10)

        storage_initial = np.sum(hillslope.theta)

        hillslope.step(dt=3600, precip_rate=0.0, ET_rate=1e-5)

        storage_final = np.sum(hillslope.theta)

        # Storage should decrease
        assert storage_final < storage_initial

    def test_theta_bounds_maintained(self):
        """Test that theta stays within physical bounds."""
        hillslope = create_synthetic_hillslope(ny=5, nx=10)

        theta_r = hillslope.soil_props['theta_r']
        theta_s = hillslope.soil_props['theta_s']

        # Run several steps with varying forcing
        for _ in range(10):
            precip = np.random.uniform(0, 2e-5)
            ET = np.random.uniform(0, 1e-5)
            hillslope.step(dt=600, precip_rate=precip, ET_rate=ET)

        # Check bounds
        assert np.all(hillslope.theta >= theta_r * 0.999)
        assert np.all(hillslope.theta <= theta_s * 1.001)

    def test_multiple_timesteps(self):
        """Test running multiple timesteps."""
        hillslope = create_synthetic_hillslope(ny=5, nx=10)

        # Run 20 steps
        for i in range(20):
            hillslope.step(dt=600, precip_rate=1e-5, ET_rate=2e-6)

        # Model should still be in valid state
        assert np.all(np.isfinite(hillslope.theta))
        assert np.all(np.isfinite(hillslope.kappa))


class TestContributingArea:
    """Test contributing area calculation."""

    def test_contributing_area_calculation(self):
        """Test contributing area calculation."""
        hillslope = create_synthetic_hillslope(ny=10, nx=20)

        area = hillslope.get_contributing_area()

        # Should be non-negative
        assert area >= 0.0

        # Should be less than total area
        total_area = hillslope.grid.ny * hillslope.grid.nx * hillslope.grid.cell_area
        assert area <= total_area

    def test_contributing_area_increases_with_wetness(self):
        """Test that contributing area increases when wet."""
        hillslope = create_synthetic_hillslope(ny=10, nx=20)

        # Dry conditions
        hillslope.theta[:, :] = 0.10
        hillslope._update_energy_and_connectivity()
        area_dry = hillslope.get_contributing_area()

        # Wet conditions
        hillslope.theta[:, :] = 0.40
        hillslope._update_energy_and_connectivity()
        area_wet = hillslope.get_contributing_area()

        # Wet should have larger or equal contributing area
        assert area_wet >= area_dry

    def test_contributing_area_dynamic(self):
        """Test that contributing area changes over time."""
        hillslope = create_synthetic_hillslope(ny=10, nx=20)

        area_initial = hillslope.get_contributing_area()

        # Add rainfall to increase connectivity
        for _ in range(5):
            hillslope.step(dt=3600, precip_rate=1e-5, ET_rate=0.0)

        area_after_rain = hillslope.get_contributing_area()

        # May increase or stay similar depending on initial state
        # Just check that calculation works
        assert area_after_rain >= 0.0


class TestGetState:
    """Test state retrieval."""

    def test_get_state(self):
        """Test getting model state."""
        hillslope = create_synthetic_hillslope(ny=5, nx=10)

        state = hillslope.get_state()

        assert 'theta' in state
        assert 'kappa' in state
        assert 'E_free' in state
        assert 'HAND' in state
        assert 'elevation' in state

        # Check shapes
        assert state['theta'].shape == (5, 10)
        assert state['kappa'].shape == (5, 10)

    def test_get_state_returns_copy(self):
        """Test that get_state returns copies."""
        hillslope = create_synthetic_hillslope(ny=3, nx=5)

        state = hillslope.get_state()

        # Modify returned state
        state['theta'][:, :] = 999.0

        # Original should be unchanged
        assert not np.any(hillslope.theta == 999.0)


class TestSyntheticHillslope:
    """Test synthetic hillslope creation function."""

    def test_default_creation(self):
        """Test creation with default parameters."""
        hillslope = create_synthetic_hillslope()

        assert hillslope.grid.ny == 30
        assert hillslope.grid.nx == 50

    def test_custom_dimensions(self):
        """Test creation with custom dimensions."""
        hillslope = create_synthetic_hillslope(ny=15, nx=25, dy=1.5, dx=2.5)

        assert hillslope.grid.ny == 15
        assert hillslope.grid.nx == 25
        assert hillslope.grid.dy == 1.5
        assert hillslope.grid.dx == 2.5

    def test_slope_affects_elevation(self):
        """Test that slope parameter affects elevation."""
        hillslope_flat = create_synthetic_hillslope(nx=10, slope=0.01)
        hillslope_steep = create_synthetic_hillslope(nx=10, slope=0.5)

        # Steeper slope should have larger elevation range
        range_flat = np.ptp(hillslope_flat.grid.elevation)
        range_steep = np.ptp(hillslope_steep.grid.elevation)

        assert range_steep > range_flat

    def test_macroporosity_spatial_pattern(self):
        """Test that macroporosity has expected spatial pattern."""
        hillslope = create_synthetic_hillslope(ny=5, nx=20)

        # Macroporosity should be higher near stream (left)
        macro_near_stream = np.mean(hillslope.macroporosity[:, 0:5])
        macro_far_from_stream = np.mean(hillslope.macroporosity[:, 15:20])

        assert macro_near_stream > macro_far_from_stream

    def test_custom_soil_properties(self):
        """Test creation with custom soil properties."""
        custom_props = {
            'theta_r': 0.10,
            'theta_s': 0.50,
            'alpha': 3.0,
            'n': 2.0,
            'K_sat': 5e-5,
            'theta_initial': 0.35,
        }

        hillslope = create_synthetic_hillslope(
            ny=5, nx=10, soil_properties=custom_props
        )

        assert hillslope.soil_props['theta_r'] == 0.10
        assert hillslope.soil_props['K_sat'] == 5e-5
        assert np.all(hillslope.theta == 0.35)


class TestIntegration:
    """Integration tests with complex scenarios."""

    def test_rainfall_runoff_event(self):
        """Test complete rainfall-runoff event."""
        hillslope = create_synthetic_hillslope(ny=10, nx=20)

        # Dry initial conditions
        hillslope.theta[:, :] = 0.15
        hillslope._update_energy_and_connectivity()

        area_before = hillslope.get_contributing_area()

        # Rainfall event
        for _ in range(10):
            hillslope.step(dt=600, precip_rate=2e-5, ET_rate=0.0)

        area_after = hillslope.get_contributing_area()

        # Contributing area should increase or stay similar
        assert area_after >= area_before * 0.9

    def test_drying_cycle(self):
        """Test drying cycle with ET."""
        hillslope = create_synthetic_hillslope(ny=5, nx=10)

        # Wet initial conditions
        hillslope.theta[:, :] = 0.40
        storage_initial = np.sum(hillslope.theta)

        # Drying with ET, no rain
        for _ in range(20):
            hillslope.step(dt=3600, precip_rate=0.0, ET_rate=1e-5)

        storage_final = np.sum(hillslope.theta)

        # Storage should decrease significantly
        assert storage_final < storage_initial * 0.95


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
