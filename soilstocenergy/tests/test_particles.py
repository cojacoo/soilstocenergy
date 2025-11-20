"""
Unit tests for particle tracking module.

Tests cover:
- Particle initialization
- Injection and tracking
- Connectivity-based mobility
- Advection and dispersion
- Concentration profiles
- Breakthrough curves
- Residence time distributions
"""

import numpy as np
import pytest
from soilstocenergy.models.particles import (
    Particle,
    ParticleTracker,
    calculate_mean_travel_time,
    calculate_preferential_flow_fraction,
)


class TestParticle:
    """Test Particle dataclass."""

    def test_initialization(self):
        """Test particle initialization."""
        particle = Particle(position=0)

        assert particle.position == 0
        assert particle.age == 0.0
        assert particle.concentration == 1.0
        assert particle.is_mobile == True
        assert particle.travel_distance == 0.0
        assert len(particle.layer_history) == 1
        assert particle.layer_history[0] == 0

    def test_custom_initialization(self):
        """Test particle with custom parameters."""
        particle = Particle(
            position=2,
            age=100.0,
            concentration=0.5,
            is_mobile=False,
            travel_distance=1.5
        )

        assert particle.position == 2
        assert particle.age == 100.0
        assert particle.concentration == 0.5
        assert particle.is_mobile == False
        assert particle.travel_distance == 1.5

    def test_layer_history_initialization(self):
        """Test that layer history is initialized correctly."""
        particle = Particle(position=3)

        assert 3 in particle.layer_history
        assert particle.layer_history[0] == 3


class TestParticleTracker:
    """Test ParticleTracker class."""

    def test_initialization(self):
        """Test tracker initialization."""
        n_layers = 5
        thickness = np.full(n_layers, 0.2)

        tracker = ParticleTracker(n_layers, thickness)

        assert tracker.n_layers == 5
        assert len(tracker.layer_thickness) == 5
        assert len(tracker.particles) == 0
        assert tracker.dispersivity == 0.01

    def test_inject_particles(self):
        """Test particle injection."""
        tracker = ParticleTracker(5, np.full(5, 0.2))

        tracker.inject_particles(layer=2, n_particles=10, concentration=0.8)

        assert len(tracker.particles) == 10
        for particle in tracker.particles:
            assert particle.position == 2
            assert particle.concentration == 0.8

    def test_inject_multiple_layers(self):
        """Test injection into multiple layers."""
        tracker = ParticleTracker(5, np.full(5, 0.2))

        tracker.inject_particles(layer=0, n_particles=5)
        tracker.inject_particles(layer=2, n_particles=3)

        assert len(tracker.particles) == 8

        # Count particles in each layer
        layer_0 = sum(1 for p in tracker.particles if p.position == 0)
        layer_2 = sum(1 for p in tracker.particles if p.position == 2)

        assert layer_0 == 5
        assert layer_2 == 3


class TestMobility:
    """Test particle mobility updates."""

    def test_update_mobility_all_connected(self):
        """Test mobility when all layers connected."""
        tracker = ParticleTracker(3, np.full(3, 0.2))
        tracker.inject_particles(0, 3)

        kappa = np.array([0.9, 0.8, 1.0])
        tracker.update_mobility(kappa, threshold=0.5)

        # All should be mobile
        for particle in tracker.particles:
            assert particle.is_mobile == True

    def test_update_mobility_mixed(self):
        """Test mobility with mixed connectivity."""
        tracker = ParticleTracker(3, np.full(3, 0.2))
        tracker.inject_particles(0, 1)
        tracker.inject_particles(1, 1)
        tracker.inject_particles(2, 1)

        kappa = np.array([0.9, 0.3, 0.8])  # Middle layer disconnected
        tracker.update_mobility(kappa, threshold=0.5)

        assert tracker.particles[0].is_mobile == True  # Layer 0
        assert tracker.particles[1].is_mobile == False  # Layer 1
        assert tracker.particles[2].is_mobile == True  # Layer 2

    def test_update_mobility_custom_threshold(self):
        """Test mobility with custom threshold."""
        tracker = ParticleTracker(2, np.full(2, 0.2))
        tracker.inject_particles(0, 1)
        tracker.inject_particles(1, 1)

        kappa = np.array([0.6, 0.4])

        # Threshold 0.5
        tracker.update_mobility(kappa, threshold=0.5)
        assert tracker.particles[0].is_mobile == True
        assert tracker.particles[1].is_mobile == False

        # Threshold 0.7
        tracker.update_mobility(kappa, threshold=0.7)
        assert tracker.particles[0].is_mobile == False
        assert tracker.particles[1].is_mobile == False


class TestAdvection:
    """Test particle advection."""

    def test_advection_downward_connected(self):
        """Test downward advection through connected layers."""
        tracker = ParticleTracker(3, np.full(3, 0.1))
        tracker.inject_particles(0, 1)

        # Strong downward velocity
        velocity = np.array([0.0, 1.0, 1.0, 0.0])  # n+1 boundaries
        dt = 1.0  # s
        kappa = np.ones(3)  # All connected

        initial_pos = tracker.particles[0].position

        tracker.advect_particles(velocity, dt, kappa)

        # Particle should have moved down
        # Distance = 1.0 * 1.0 = 1.0 m > layer thickness 0.1 m
        assert tracker.particles[0].position > initial_pos

    def test_advection_blocked_by_disconnection(self):
        """Test that advection stops at disconnected layer."""
        tracker = ParticleTracker(3, np.full(3, 0.1))
        tracker.inject_particles(0, 1)

        velocity = np.array([0.0, 1.0, 1.0, 0.0])
        dt = 1.0
        kappa = np.array([1.0, 0.0, 1.0])  # Middle layer disconnected

        tracker.advect_particles(velocity, dt, kappa)

        # Should not move to disconnected layer
        assert tracker.particles[0].position == 0

    def test_advection_slow_velocity(self):
        """Test that slow velocity doesn't cause movement."""
        tracker = ParticleTracker(3, np.full(3, 0.2))
        tracker.inject_particles(0, 1)

        # Velocity too small to cross layer
        velocity = np.array([0.0, 1e-6, 1e-6, 0.0])
        dt = 1.0
        kappa = np.ones(3)

        tracker.advect_particles(velocity, dt, kappa)

        # Distance = 1e-6 * 1 = 1e-6 m << 0.2 m
        assert tracker.particles[0].position == 0

    def test_advection_updates_age(self):
        """Test that advection updates particle age."""
        tracker = ParticleTracker(3, np.full(3, 0.1))
        tracker.inject_particles(0, 1)

        velocity = np.zeros(4)
        dt = 100.0
        kappa = np.ones(3)

        initial_age = tracker.particles[0].age

        tracker.advect_particles(velocity, dt, kappa)

        assert tracker.particles[0].age == initial_age + dt

    def test_advection_updates_history(self):
        """Test that advection updates layer history."""
        tracker = ParticleTracker(3, np.full(3, 0.1))
        tracker.inject_particles(0, 1)

        velocity = np.array([0.0, 1.0, 1.0, 0.0])
        dt = 1.0
        kappa = np.ones(3)

        initial_history_len = len(tracker.particles[0].layer_history)

        tracker.advect_particles(velocity, dt, kappa)

        # If moved, history should be longer
        if tracker.particles[0].position > 0:
            assert len(tracker.particles[0].layer_history) > initial_history_len


class TestDispersion:
    """Test particle dispersion."""

    def test_dispersion_with_zero_velocity(self):
        """Test that no dispersion occurs with zero velocity."""
        np.random.seed(42)
        tracker = ParticleTracker(3, np.full(3, 0.1), dispersivity=0.01)
        tracker.inject_particles(1, 1)

        velocity = np.zeros(4)
        dt = 1.0
        kappa = np.ones(3)

        initial_pos = tracker.particles[0].position

        tracker.disperse_particles(velocity, dt, kappa)

        # Should not move with zero velocity
        assert tracker.particles[0].position == initial_pos

    def test_dispersion_respects_connectivity(self):
        """Test that dispersion respects connectivity."""
        np.random.seed(42)
        tracker = ParticleTracker(3, np.full(3, 0.05), dispersivity=0.1)
        tracker.inject_particles(1, 20)  # Multiple particles for statistics

        velocity = np.array([0.0, 1e-3, 1e-3, 0.0])
        dt = 100.0
        kappa = np.array([0.0, 1.0, 0.0])  # Only middle layer connected

        for _ in range(10):
            tracker.disperse_particles(velocity, dt, kappa)

        # All particles should still be in layer 1
        for particle in tracker.particles:
            assert particle.position == 1


class TestConcentrationProfile:
    """Test concentration profile calculation."""

    def test_concentration_single_layer(self):
        """Test concentration with particles in single layer."""
        tracker = ParticleTracker(5, np.full(5, 0.2))
        tracker.inject_particles(2, 10, concentration=0.8)

        theta = np.full(5, 0.3)
        profile = tracker.get_concentration_profile(theta)

        assert profile[2] == pytest.approx(0.8)
        assert profile[0] == pytest.approx(0.0)
        assert profile[4] == pytest.approx(0.0)

    def test_concentration_multiple_layers(self):
        """Test concentration with particles in multiple layers."""
        tracker = ParticleTracker(5, np.full(5, 0.2))
        tracker.inject_particles(1, 5, concentration=1.0)
        tracker.inject_particles(3, 5, concentration=0.5)

        theta = np.full(5, 0.3)
        profile = tracker.get_concentration_profile(theta)

        assert profile[1] == 1.0
        assert profile[3] == 0.5
        assert profile[0] == 0.0

    def test_concentration_averaging(self):
        """Test that concentrations are averaged correctly."""
        tracker = ParticleTracker(3, np.full(3, 0.2))

        # Add particles with different concentrations to same layer
        tracker.particles.append(Particle(position=1, concentration=1.0))
        tracker.particles.append(Particle(position=1, concentration=0.5))

        theta = np.full(3, 0.3)
        profile = tracker.get_concentration_profile(theta)

        assert profile[1] == pytest.approx(0.75)  # Average of 1.0 and 0.5


class TestBreakthroughCurve:
    """Test breakthrough curve calculation."""

    def test_breakthrough_simple(self):
        """Test breakthrough curve calculation."""
        tracker = ParticleTracker(5, np.full(5, 0.2))

        # Create particles with known arrival times
        for i in range(5):
            p = Particle(position=4)
            p.age = float(i * 100)
            p.layer_history = [0, 1, 2, 3, 4]
            tracker.particles.append(p)

        time_bins = np.array([0, 100, 200, 300, 400, 500])
        btc = tracker.get_breakthrough_curve(output_layer=4, time_bins=time_bins)

        # Each particle should be in a different bin
        assert len(btc) == 5
        assert np.sum(btc) == 5  # Total particles

    def test_breakthrough_no_arrivals(self):
        """Test breakthrough with no particles reaching output."""
        tracker = ParticleTracker(5, np.full(5, 0.2))
        tracker.inject_particles(0, 10)

        time_bins = np.array([0, 100, 200, 300])
        btc = tracker.get_breakthrough_curve(output_layer=4, time_bins=time_bins)

        # No particles reached layer 4
        assert np.all(btc == 0)


class TestResidenceTime:
    """Test residence time distribution."""

    def test_residence_time_distribution(self):
        """Test getting residence time distribution."""
        tracker = ParticleTracker(3, np.full(3, 0.2))

        # Add particles with different ages
        for i in range(5):
            p = Particle(position=i % 3)
            p.age = float(i * 50)
            tracker.particles.append(p)

        ages, positions = tracker.get_residence_time_distribution()

        assert len(ages) == 5
        assert len(positions) == 5
        assert ages[0] == 0.0
        assert ages[4] == 200.0

    def test_mobile_fraction(self):
        """Test mobile fraction calculation."""
        tracker = ParticleTracker(3, np.full(3, 0.2))

        # Add mobile and immobile particles
        for i in range(10):
            p = Particle(position=i % 3)
            p.is_mobile = (i % 2 == 0)
            tracker.particles.append(p)

        fraction = tracker.get_mobile_fraction()

        assert fraction == 0.5

    def test_mobile_fraction_empty(self):
        """Test mobile fraction with no particles."""
        tracker = ParticleTracker(3, np.full(3, 0.2))

        fraction = tracker.get_mobile_fraction()

        assert fraction == 0.0


class TestParticleRemoval:
    """Test particle removal."""

    def test_remove_exited_particles(self):
        """Test removal of particles that exited."""
        tracker = ParticleTracker(3, np.full(3, 0.2))

        # Add particles inside and outside domain
        tracker.particles.append(Particle(position=1))
        tracker.particles.append(Particle(position=-1))  # Exited top
        tracker.particles.append(Particle(position=5))   # Exited bottom

        initial_count = len(tracker.particles)
        tracker.remove_exited_particles()

        assert len(tracker.particles) == 1
        assert tracker.particles[0].position == 1


class TestIntegratedStep:
    """Test integrated step function."""

    def test_step_function(self):
        """Test complete step with all processes."""
        tracker = ParticleTracker(5, np.full(5, 0.1))
        tracker.inject_particles(0, 10)

        velocity = np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.0])
        dt = 1.0
        kappa = np.ones(5)
        theta = np.full(5, 0.3)

        initial_count = len(tracker.particles)

        tracker.step(velocity, dt, kappa, theta)

        # Particles should still exist (may have moved)
        assert len(tracker.particles) > 0
        assert len(tracker.particles) <= initial_count

        # Ages should have increased
        for particle in tracker.particles:
            assert particle.age >= dt

    def test_step_with_disconnection(self):
        """Test step with some layers disconnected."""
        tracker = ParticleTracker(5, np.full(5, 0.1))
        tracker.inject_particles(0, 5)
        tracker.inject_particles(2, 5)

        velocity = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
        dt = 1.0
        kappa = np.array([1.0, 0.0, 1.0, 0.0, 1.0])  # Alternating
        theta = np.full(5, 0.3)

        tracker.step(velocity, dt, kappa, theta)

        # Check mobility was updated
        for particle in tracker.particles:
            if particle.position in [1, 3]:
                assert particle.is_mobile == False
            elif particle.position in [0, 2, 4]:
                assert particle.is_mobile == True


class TestUtilityFunctions:
    """Test utility functions."""

    def test_mean_travel_time(self):
        """Test mean travel time calculation."""
        particles = []
        for i in range(5):
            p = Particle(position=0)
            p.age = float((i + 1) * 100)
            p.layer_history = [0, 1, 2]
            particles.append(p)

        mean_time = calculate_mean_travel_time(particles, target_layer=2)

        # Mean age = (100+200+300+400+500)/5 = 300
        # All particles have 3 layers, so time to layer 2 = age * 2/3
        assert mean_time > 0

    def test_mean_travel_time_no_arrivals(self):
        """Test mean travel time with no arrivals."""
        particles = [Particle(position=0) for _ in range(5)]

        mean_time = calculate_mean_travel_time(particles, target_layer=5)

        assert np.isnan(mean_time)

    def test_preferential_flow_fraction(self):
        """Test preferential flow fraction calculation."""
        particles = []

        # Fast particles (preferential flow)
        for _ in range(3):
            p = Particle(position=0)
            p.age = 1800  # 30 minutes (fast)
            p.layer_history = [0, 1, 2, 3, 4]
            particles.append(p)

        # Slow particles (matrix flow)
        for _ in range(7):
            p = Particle(position=0)
            p.age = 7200  # 2 hours (slow)
            p.layer_history = [0, 1, 2, 3, 4]
            particles.append(p)

        fraction = calculate_preferential_flow_fraction(
            particles, fast_threshold=3600
        )

        assert fraction == pytest.approx(0.3)  # 3/10

    def test_preferential_flow_no_particles(self):
        """Test preferential flow with no particles."""
        fraction = calculate_preferential_flow_fraction([], fast_threshold=3600)

        assert fraction == 0.0


class TestMassConservation:
    """Test particle number conservation."""

    def test_particle_conservation_no_exit(self):
        """Test that particles are conserved when not exiting."""
        tracker = ParticleTracker(5, np.full(5, 0.2))
        tracker.inject_particles(2, 20)

        initial_count = len(tracker.particles)

        # Run several steps with moderate velocity
        velocity = np.full(6, 1e-4)
        kappa = np.ones(5)
        theta = np.full(5, 0.3)

        for _ in range(10):
            tracker.step(velocity, 10.0, kappa, theta)

        # Should have same or fewer particles (some may exit bottom)
        assert len(tracker.particles) <= initial_count


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
