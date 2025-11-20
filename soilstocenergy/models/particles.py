"""
Particle tracking for water and solute transport.

This module implements particle-based tracking inspired by echoRD for
following water movement and solute transport through dynamically
connected soil networks.

Key features:
- Particle representation of water parcels
- Advection through connected paths only
- Immobilization in disconnected regions
- Tracer concentration tracking
- Age and residence time distributions
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Particle:
    """
    A water particle carrying tracer/solute.

    Attributes
    ----------
    position : int
        Current layer index
    age : float
        Particle age since injection [s]
    concentration : float
        Tracer concentration [kg/m³]
    is_mobile : bool
        Whether particle can move (connected)
    travel_distance : float
        Total distance traveled [m]
    layer_history : list
        History of layers visited
    """
    position: int
    age: float = 0.0
    concentration: float = 1.0
    is_mobile: bool = True
    travel_distance: float = 0.0
    layer_history: List[int] = field(default_factory=list)

    def __post_init__(self):
        """Initialize history with starting position."""
        if not self.layer_history:
            self.layer_history.append(self.position)


class ParticleTracker:
    """
    Track particles through 1D soil column with dynamic connectivity.

    Particles move through connected layers via advection and dispersion.
    Immobilized in disconnected regions until reconnection.
    """

    def __init__(
        self,
        n_layers: int,
        layer_thickness: np.ndarray,
        dispersivity: float = 0.01
    ):
        """
        Initialize particle tracker.

        Parameters
        ----------
        n_layers : int
            Number of soil layers
        layer_thickness : array
            Thickness of each layer [m]
        dispersivity : float
            Longitudinal dispersivity [m]
        """
        self.n_layers = n_layers
        self.layer_thickness = layer_thickness
        self.dispersivity = dispersivity
        self.particles: List[Particle] = []

    def inject_particles(
        self,
        layer: int,
        n_particles: int,
        concentration: float = 1.0
    ):
        """
        Inject particles into specified layer.

        Parameters
        ----------
        layer : int
            Layer index for injection
        n_particles : int
            Number of particles to inject
        concentration : float
            Initial tracer concentration [kg/m³]
        """
        for _ in range(n_particles):
            particle = Particle(
                position=layer,
                concentration=concentration
            )
            self.particles.append(particle)

    def update_mobility(self, kappa: np.ndarray, threshold: float = 0.5):
        """
        Update particle mobility based on connectivity.

        Parameters
        ----------
        kappa : array
            Connectivity field [-]
        threshold : float
            Threshold for mobility
        """
        for particle in self.particles:
            if 0 <= particle.position < self.n_layers:
                particle.is_mobile = kappa[particle.position] > threshold

    def advect_particles(
        self,
        velocity: np.ndarray,
        dt: float,
        kappa: np.ndarray,
        threshold: float = 0.5
    ):
        """
        Advect particles through connected network.

        Particles move only if:
        1. Current layer is connected (κ > threshold)
        2. Target layer is connected
        3. Velocity carries them to next layer

        Parameters
        ----------
        velocity : array
            Vertical velocity at layer boundaries [m/s]
            Positive = downward
        dt : float
            Time step [s]
        kappa : array
            Connectivity field [-]
        threshold : float
            Connectivity threshold for movement
        """
        for particle in self.particles:
            # Update age
            particle.age += dt

            # Skip if not mobile
            if not particle.is_mobile:
                continue

            # Skip if at boundaries
            if particle.position < 0 or particle.position >= self.n_layers:
                continue

            # Get velocity at layer bottom
            v = velocity[particle.position + 1] if particle.position < self.n_layers - 1 else 0.0

            # Distance traveled this step
            distance = abs(v) * dt

            # Downward movement
            if v > 0 and particle.position < self.n_layers - 1:
                # Check if next layer is connected
                if kappa[particle.position + 1] > threshold:
                    # Move if distance exceeds layer thickness
                    if distance >= self.layer_thickness[particle.position]:
                        particle.position += 1
                        particle.travel_distance += self.layer_thickness[particle.position - 1]
                        particle.layer_history.append(particle.position)

            # Upward movement (less common but possible)
            elif v < 0 and particle.position > 0:
                if kappa[particle.position - 1] > threshold:
                    if distance >= self.layer_thickness[particle.position]:
                        particle.position -= 1
                        particle.travel_distance += self.layer_thickness[particle.position + 1]
                        particle.layer_history.append(particle.position)

    def disperse_particles(
        self,
        velocity: np.ndarray,
        dt: float,
        kappa: np.ndarray,
        threshold: float = 0.5
    ):
        """
        Apply dispersion to particle movement.

        Dispersion causes random displacement proportional to dispersivity.

        Parameters
        ----------
        velocity : array
            Vertical velocity [m/s]
        dt : float
            Time step [s]
        kappa : array
            Connectivity field [-]
        threshold : float
            Connectivity threshold
        """
        for particle in self.particles:
            if not particle.is_mobile:
                continue

            if particle.position < 0 or particle.position >= self.n_layers:
                continue

            # Get local velocity
            v = abs(velocity[particle.position + 1]) if particle.position < self.n_layers - 1 else 0.0

            if v < 1e-10:
                continue

            # Dispersive displacement
            D = self.dispersivity * v
            sigma = np.sqrt(2 * D * dt)
            displacement = np.random.normal(0, sigma)

            # Move up or down based on random displacement
            if abs(displacement) > self.layer_thickness[particle.position] / 2:
                direction = 1 if displacement > 0 else -1
                new_pos = particle.position + direction

                # Check bounds and connectivity
                if 0 <= new_pos < self.n_layers:
                    if kappa[new_pos] > threshold:
                        particle.position = new_pos
                        particle.layer_history.append(new_pos)

    def remove_exited_particles(self):
        """Remove particles that have exited the column."""
        self.particles = [p for p in self.particles
                         if 0 <= p.position < self.n_layers]

    def get_concentration_profile(self, theta: np.ndarray) -> np.ndarray:
        """
        Calculate concentration profile from particle distribution.

        Parameters
        ----------
        theta : array
            Water content in each layer [-]

        Returns
        -------
        array
            Concentration in each layer [kg/m³]
        """
        concentration = np.zeros(self.n_layers)
        particle_count = np.zeros(self.n_layers)

        for particle in self.particles:
            if 0 <= particle.position < self.n_layers:
                concentration[particle.position] += particle.concentration
                particle_count[particle.position] += 1

        # Average concentration per layer
        mask = particle_count > 0
        concentration[mask] /= particle_count[mask]

        return concentration

    def get_breakthrough_curve(
        self,
        output_layer: int,
        time_bins: np.ndarray
    ) -> np.ndarray:
        """
        Calculate breakthrough curve at specified layer.

        Parameters
        ----------
        output_layer : int
            Layer index for breakthrough monitoring
        time_bins : array
            Time bins for breakthrough curve [s]

        Returns
        -------
        array
            Particle count in each time bin
        """
        # Find particles that reached output layer
        ages = []
        for particle in self.particles:
            if output_layer in particle.layer_history:
                # Find when particle first reached this layer
                idx = particle.layer_history.index(output_layer)
                # Approximate age when reached (assume linear)
                age_at_arrival = particle.age * idx / len(particle.layer_history)
                ages.append(age_at_arrival)

        if not ages:
            return np.zeros(len(time_bins) - 1)

        # Histogram of arrival times
        counts, _ = np.histogram(ages, bins=time_bins)

        return counts

    def get_residence_time_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get residence time distribution for all particles.

        Returns
        -------
        ages : array
            Particle ages [s]
        positions : array
            Current positions
        """
        ages = np.array([p.age for p in self.particles])
        positions = np.array([p.position for p in self.particles])

        return ages, positions

    def get_mobile_fraction(self) -> float:
        """
        Calculate fraction of mobile particles.

        Returns
        -------
        float
            Fraction of particles that can move
        """
        if len(self.particles) == 0:
            return 0.0

        mobile_count = sum(1 for p in self.particles if p.is_mobile)
        return mobile_count / len(self.particles)

    def step(
        self,
        velocity: np.ndarray,
        dt: float,
        kappa: np.ndarray,
        theta: np.ndarray,
        threshold: float = 0.5
    ):
        """
        Advance particle tracker by one time step.

        Parameters
        ----------
        velocity : array
            Vertical velocity field [m/s]
        dt : float
            Time step [s]
        kappa : array
            Connectivity field [-]
        theta : array
            Water content field [-]
        threshold : float
            Connectivity threshold
        """
        # Update mobility based on current connectivity
        self.update_mobility(kappa, threshold)

        # Advect particles
        self.advect_particles(velocity, dt, kappa, threshold)

        # Apply dispersion
        self.disperse_particles(velocity, dt, kappa, threshold)

        # Remove particles that exited
        self.remove_exited_particles()


def calculate_mean_travel_time(particles: List[Particle], target_layer: int) -> float:
    """
    Calculate mean travel time to target layer.

    Parameters
    ----------
    particles : list
        List of particles
    target_layer : int
        Target layer index

    Returns
    -------
    float
        Mean travel time [s]
    """
    travel_times = []

    for particle in particles:
        if target_layer in particle.layer_history:
            # Approximate time based on position in history
            idx = particle.layer_history.index(target_layer)
            travel_time = particle.age * idx / len(particle.layer_history)
            travel_times.append(travel_time)

    if not travel_times:
        return np.nan

    return np.mean(travel_times)


def calculate_preferential_flow_fraction(
    particles: List[Particle],
    fast_threshold: float = 3600.0
) -> float:
    """
    Calculate fraction undergoing preferential flow.

    Particles reaching bottom quickly (< threshold) are considered
    preferential flow.

    Parameters
    ----------
    particles : list
        List of particles
    fast_threshold : float
        Time threshold for "fast" flow [s]

    Returns
    -------
    float
        Fraction of particles in preferential flow
    """
    if not particles:
        return 0.0

    # Count particles that reached bottom layer quickly
    fast_count = 0
    reached_bottom = 0

    for particle in particles:
        # Check if reached bottom (last layer)
        if particle.layer_history and max(particle.layer_history) >= len(particle.layer_history) - 1:
            reached_bottom += 1
            if particle.age < fast_threshold:
                fast_count += 1

    if reached_bottom == 0:
        return 0.0

    return fast_count / reached_bottom
