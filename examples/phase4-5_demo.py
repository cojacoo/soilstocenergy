"""
Demonstration of Phase 4-5: Percolation Network and 1D Vertical Model

This example demonstrates:
1. Percolation network analysis (cluster identification, percolation thresholds)
2. 1D soil column with dynamic connectivity
3. Infiltration and drainage processes
4. Connectivity effects on water movement
5. Percolation transitions
"""

import numpy as np
import matplotlib.pyplot as plt

from soilstocenergy.core.percolation import (
    PercolationNetwork,
    ScalingLaws,
    calculate_correlation_length,
)
from soilstocenergy.models.vertical_1d import (
    create_uniform_column,
    create_layered_column,
)


def demo_percolation_network():
    """Demonstrate percolation network analysis."""
    print("=" * 60)
    print("Demo 1: Percolation Network Analysis")
    print("=" * 60)

    # Create 2D network
    network = PercolationNetwork(shape=(20, 20))

    # Test different occupation probabilities
    p_values = np.linspace(0.3, 0.9, 20)
    percolating = []
    largest_cluster_sizes = []
    active_fractions = []

    for p in p_values:
        # Random connectivity field
        kappa = np.random.rand(20, 20)
        threshold = 1 - p  # Convert p to threshold
        network.update_active_state(kappa, threshold=threshold)

        # Check percolation
        perc = network.check_percolation(direction='vertical')
        percolating.append(perc)

        # Largest cluster
        _, size = network.get_largest_cluster()
        largest_cluster_sizes.append(size)

        # Active fraction
        active_fractions.append(network.calculate_active_fraction())

    percolating = np.array(percolating)

    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Percolation probability
    axes[0].plot(p_values, percolating.astype(float), 'bo-', linewidth=2)
    axes[0].axvline(0.5927, color='r', linestyle='--',
                    label=f"Theoretical p_c = {ScalingLaws.P_C['2d_square']:.3f}")
    axes[0].set_xlabel('Occupation Probability p')
    axes[0].set_ylabel('Percolation Probability')
    axes[0].set_title('Percolation Transition in 2D Network')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Largest cluster size
    axes[1].plot(p_values, np.array(largest_cluster_sizes) / 400,
                'gs-', linewidth=2)
    axes[1].axvline(0.5927, color='r', linestyle='--',
                    label='Theoretical p_c')
    axes[1].set_xlabel('Occupation Probability p')
    axes[1].set_ylabel('Normalized Largest Cluster Size')
    axes[1].set_title('Cluster Size Near Percolation Threshold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/soilstocenergy/examples/percolation_network.png',
                dpi=150, bbox_inches='tight')
    print("✓ Percolation network plot saved")

    # Estimate percolation threshold
    p_c_estimated = ScalingLaws.estimate_percolation_threshold(p_values, percolating)
    p_c_theoretical = ScalingLaws.P_C['2d_square']

    print(f"\nPercolation Threshold Analysis:")
    print(f"  Estimated p_c: {p_c_estimated:.3f}")
    print(f"  Theoretical p_c: {p_c_theoretical:.3f}")
    print(f"  Difference: {abs(p_c_estimated - p_c_theoretical):.3f}")
    print()


def demo_1d_column_basic():
    """Demonstrate basic 1D soil column."""
    print("=" * 60)
    print("Demo 2: 1D Soil Column - Basic Water Balance")
    print("=" * 60)

    # Create uniform column
    column = create_uniform_column(
        n_layers=10,
        total_depth=1.0,
        theta_r=0.05,
        theta_s=0.45,
        alpha=2.0,
        n=1.5,
        K_sat=1e-5,
        E_crit=-5000.0  # Moderate threshold
    )

    # Simulate rainfall event
    dt = 600  # 10 minutes
    precip_rate = 5e-5  # 5 cm/hr
    n_steps = 60  # 10 hours

    # Storage data
    time = []
    storage = []
    theta_profiles = []

    for step in range(n_steps):
        time.append(step * dt / 3600)  # Convert to hours
        storage.append(column.get_total_storage())

        # Store profile every 10 steps
        if step % 10 == 0:
            theta_profiles.append(column.theta.copy())

        # Apply precipitation for first 3 hours
        if step < 18:
            column.step(dt, precip_rate=precip_rate, ET_rate=0.0)
        else:
            column.step(dt, precip_rate=0.0, ET_rate=0.0)

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Storage over time
    axes[0].plot(time, storage, 'b-', linewidth=2)
    axes[0].axvline(3, color='r', linestyle='--', alpha=0.5,
                    label='Rain stops')
    axes[0].set_xlabel('Time [hours]')
    axes[0].set_ylabel('Total Storage [m]')
    axes[0].set_title('Water Storage During Infiltration Event')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Moisture profiles
    profile = column.get_profile()
    depths = profile['depth']

    for i, theta_prof in enumerate(theta_profiles):
        axes[1].plot(theta_prof, depths, label=f't = {i} hr')

    axes[1].set_xlabel('Water Content θ [-]')
    axes[1].set_ylabel('Depth [m]')
    axes[1].set_title('Moisture Profiles Over Time')
    axes[1].invert_yaxis()
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/soilstocenergy/examples/1d_column_basic.png',
                dpi=150, bbox_inches='tight')
    print("✓ 1D column basic plot saved")
    print()


def demo_connectivity_effects():
    """Demonstrate connectivity effects on water movement."""
    print("=" * 60)
    print("Demo 3: Connectivity Effects on Water Movement")
    print("=" * 60)

    # Create two columns: one with low E_crit (easy connection),
    # one with high E_crit (hard connection)
    column_connected = create_uniform_column(
        n_layers=10,
        total_depth=1.0,
        E_crit=-10000.0,  # Very low -> always connected
        K_sat=1e-5
    )

    column_disconnected = create_uniform_column(
        n_layers=10,
        total_depth=1.0,
        E_crit=-500.0,  # Higher -> sometimes disconnected
        K_sat=1e-5
    )

    # Simulate same rainfall event on both
    dt = 600
    precip_rate = 5e-5
    n_steps = 60

    storage_connected = []
    storage_disconnected = []
    kappa_mean_disconnected = []

    for step in range(n_steps):
        # Apply same forcing to both
        if step < 18:
            column_connected.step(dt, precip_rate=precip_rate, ET_rate=0.0)
            column_disconnected.step(dt, precip_rate=precip_rate, ET_rate=0.0)
        else:
            column_connected.step(dt, precip_rate=0.0, ET_rate=0.0)
            column_disconnected.step(dt, precip_rate=0.0, ET_rate=0.0)

        storage_connected.append(column_connected.get_total_storage())
        storage_disconnected.append(column_disconnected.get_total_storage())
        kappa_mean_disconnected.append(np.mean(column_disconnected.kappa))

    time = np.arange(n_steps) * dt / 3600

    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Storage comparison
    axes[0].plot(time, storage_connected, 'b-', linewidth=2,
                label='Always Connected (E_crit = -10000)')
    axes[0].plot(time, storage_disconnected, 'r-', linewidth=2,
                label='Sometimes Disconnected (E_crit = -500)')
    axes[0].axvline(3, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Time [hours]')
    axes[0].set_ylabel('Total Storage [m]')
    axes[0].set_title('Effect of Connectivity on Water Storage')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Mean connectivity
    axes[1].plot(time, kappa_mean_disconnected, 'r-', linewidth=2)
    axes[1].axhline(0.5, color='gray', linestyle=':', alpha=0.5,
                    label='Threshold')
    axes[1].set_xlabel('Time [hours]')
    axes[1].set_ylabel('Mean Connectivity κ [-]')
    axes[1].set_title('Connectivity State Evolution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/soilstocenergy/examples/connectivity_effects.png',
                dpi=150, bbox_inches='tight')
    print("✓ Connectivity effects plot saved")

    print(f"\nFinal Storage:")
    print(f"  Always connected: {storage_connected[-1]:.4f} m")
    print(f"  Sometimes disconnected: {storage_disconnected[-1]:.4f} m")
    print(f"  Difference: {abs(storage_connected[-1] - storage_disconnected[-1]):.4f} m")
    print()


def demo_layered_column():
    """Demonstrate layered soil column."""
    print("=" * 60)
    print("Demo 4: Layered Soil Column")
    print("=" * 60)

    # Create column with three distinct layers
    layer_props = [
        {  # Top: Sandy loam (high conductivity)
            'depth_top': 0.0,
            'depth_bottom': 0.3,
            'theta_r': 0.05,
            'theta_s': 0.45,
            'alpha': 3.0,
            'n': 1.8,
            'K_sat': 5e-5,
            'macroporosity': 0.15,
            'E_crit': -2000.0
        },
        {  # Middle: Loam
            'depth_top': 0.3,
            'depth_bottom': 0.7,
            'theta_r': 0.08,
            'theta_s': 0.48,
            'alpha': 2.0,
            'n': 1.5,
            'K_sat': 1e-5,
            'macroporosity': 0.05,
            'E_crit': -3000.0
        },
        {  # Bottom: Clay (low conductivity)
            'depth_top': 0.7,
            'depth_bottom': 1.0,
            'theta_r': 0.10,
            'theta_s': 0.50,
            'alpha': 1.0,
            'n': 1.2,
            'K_sat': 1e-6,
            'macroporosity': 0.0,
            'E_crit': -4000.0
        },
    ]

    column = create_layered_column(layer_props)

    # Simulate infiltration
    dt = 600
    precip_rate = 3e-5
    n_steps = 120  # 20 hours

    profiles = []
    times = []

    for step in range(n_steps):
        if step % 12 == 0:  # Every 2 hours
            profiles.append(column.get_profile())
            times.append(step * dt / 3600)

        # Rain for 6 hours
        if step < 36:
            column.step(dt, precip_rate=precip_rate, ET_rate=0.0)
        else:
            column.step(dt, precip_rate=0.0, ET_rate=2e-6)

    # Plot layered profile evolution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Moisture profiles
    for i, (prof, t) in enumerate(zip(profiles, times)):
        axes[0].plot(prof['theta'], prof['depth'],
                    label=f't = {t:.1f} hr', marker='o', markersize=4)

    axes[0].set_xlabel('Water Content θ [-]')
    axes[0].set_ylabel('Depth [m]')
    axes[0].set_title('Moisture Profiles')
    axes[0].invert_yaxis()
    axes[0].legend(loc='best', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Connectivity profiles
    for i, (prof, t) in enumerate(zip(profiles, times)):
        axes[1].plot(prof['kappa'], prof['depth'],
                    label=f't = {t:.1f} hr', marker='s', markersize=4)

    axes[1].set_xlabel('Connectivity κ [-]')
    axes[1].set_ylabel('Depth [m]')
    axes[1].set_title('Connectivity Profiles')
    axes[1].invert_yaxis()
    axes[1].legend(loc='best', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Layer boundaries
    for props in layer_props:
        for ax in [axes[0], axes[1]]:
            ax.axhline(props['depth_bottom'], color='k',
                      linestyle='--', alpha=0.3)

    # Final state
    final_prof = column.get_profile()
    im = axes[2].scatter(final_prof['theta'], final_prof['depth'],
                        c=final_prof['kappa'], s=100, cmap='RdYlGn',
                        vmin=0, vmax=1, edgecolors='k')
    axes[2].set_xlabel('Water Content θ [-]')
    axes[2].set_ylabel('Depth [m]')
    axes[2].set_title('Final State (colored by κ)')
    axes[2].invert_yaxis()
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(im, ax=axes[2], label='Connectivity κ')

    plt.tight_layout()
    plt.savefig('/home/user/soilstocenergy/examples/layered_column.png',
                dpi=150, bbox_inches='tight')
    print("✓ Layered column plot saved")
    print()


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("SoilStocEnergy Phase 4-5 Demonstration")
    print("Percolation Network and 1D Vertical Model")
    print("=" * 60 + "\n")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Run demonstrations
    demo_percolation_network()
    demo_1d_column_basic()
    demo_connectivity_effects()
    demo_layered_column()

    print("=" * 60)
    print("All demonstrations completed successfully!")
    print("Plots saved to examples/ directory")
    print("=" * 60)


if __name__ == '__main__':
    main()
