"""
Demonstration of Phase 1-3: Thermodynamics and Connectivity Framework

This example demonstrates:
1. Free energy calculations with van Genuchten model
2. rDUNE index calculation
3. Connectivity state calculations (binary, sigmoid, linear)
4. Spatial connectivity patterns
5. Hysteresis effects
"""

import numpy as np
import matplotlib.pyplot as plt

# Import our modules
from soilstocenergy.core.thermodynamics import (
    FreeEnergyCalculator,
    LocalEquilibrium,
    calculate_rDUNE,
)
from soilstocenergy.core.connectivity import (
    ConnectivityCalculator,
    ConnectivityState,
    HysteresisConnectivity,
)


def demo_free_energy():
    """Demonstrate free energy calculations."""
    print("=" * 60)
    print("Demo 1: Free Energy Calculations")
    print("=" * 60)

    # Create calculator with van Genuchten parameters (loamy soil)
    fe_calc = FreeEnergyCalculator(
        theta_r=0.05,  # Residual water content
        theta_s=0.45,  # Saturated water content
        alpha=2.0,     # van Genuchten alpha [1/m]
        n=1.5,         # van Genuchten n [-]
        model='vanGenuchten'
    )

    # Calculate free energy for different conditions
    theta_values = np.linspace(0.05, 0.45, 50)
    HAND_values = [0.0, 1.0, 2.0, 5.0]

    plt.figure(figsize=(10, 6))

    for HAND in HAND_values:
        E_free = fe_calc.calculate_free_energy(theta_values, HAND)
        plt.plot(theta_values, E_free / 1000, label=f'HAND = {HAND} m')

    plt.xlabel('Water Content θ [-]')
    plt.ylabel('Free Energy [kJ/m³]')
    plt.title('Soil Water Free Energy vs. Water Content')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/user/soilstocenergy/examples/free_energy.png', dpi=150, bbox_inches='tight')
    print("✓ Free energy plot saved to examples/free_energy.png")

    # Print some values
    print(f"\nExample calculations:")
    print(f"  θ = 0.30, HAND = 1.0 m → E_free = {fe_calc.calculate_free_energy(0.30, 1.0):.2f} J/m³")
    print(f"  θ = 0.15, HAND = 2.0 m → E_free = {fe_calc.calculate_free_energy(0.15, 2.0):.2f} J/m³")
    print()


def demo_rdune():
    """Demonstrate rDUNE index."""
    print("=" * 60)
    print("Demo 2: rDUNE Topographic Index")
    print("=" * 60)

    # Create synthetic hillslope
    distance = np.linspace(0, 100, 100)  # Distance from stream [m]
    HAND = 0.5 * distance**0.5  # Height above stream [m]
    flow_path_length = distance  # Flow path to stream [m]

    # Calculate rDUNE
    rDUNE = calculate_rDUNE(HAND, flow_path_length)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot HAND
    axes[0].plot(distance, HAND, 'b-', linewidth=2)
    axes[0].set_xlabel('Distance from Stream [m]')
    axes[0].set_ylabel('HAND [m]')
    axes[0].set_title('Height Above Nearest Drainage')
    axes[0].grid(True, alpha=0.3)

    # Plot rDUNE
    axes[1].plot(distance, rDUNE, 'r-', linewidth=2)
    axes[1].set_xlabel('Distance from Stream [m]')
    axes[1].set_ylabel('rDUNE [-]')
    axes[1].set_title('rDUNE Index (accounts for dissipation)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/soilstocenergy/examples/rdune.png', dpi=150, bbox_inches='tight')
    print("✓ rDUNE plot saved to examples/rdune.png")
    print()


def demo_connectivity_modes():
    """Demonstrate different connectivity formulations."""
    print("=" * 60)
    print("Demo 3: Connectivity Modes (binary, sigmoid, linear)")
    print("=" * 60)

    # Create calculator
    calc = ConnectivityCalculator(transition_sharpness=10.0)

    # Energy range
    E_free = np.linspace(-300, 100, 200)
    E_crit = -100.0

    # Calculate connectivity with different modes
    kappa_binary = calc.calculate_connectivity(E_free, E_crit, mode='binary')
    kappa_sigmoid = calc.calculate_connectivity(E_free, E_crit, mode='sigmoid')
    kappa_linear = calc.calculate_connectivity(E_free, E_crit, mode='linear')

    plt.figure(figsize=(10, 6))
    plt.plot(E_free, kappa_binary, 'b-', linewidth=2, label='Binary')
    plt.plot(E_free, kappa_sigmoid, 'r-', linewidth=2, label='Sigmoid')
    plt.plot(E_free, kappa_linear, 'g-', linewidth=2, label='Linear')
    plt.axvline(E_crit, color='k', linestyle='--', alpha=0.5, label='E_crit')
    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('Free Energy E_free [J/m³]')
    plt.ylabel('Connectivity κ [-]')
    plt.title('Connectivity State Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/user/soilstocenergy/examples/connectivity_modes.png', dpi=150, bbox_inches='tight')
    print("✓ Connectivity modes plot saved to examples/connectivity_modes.png")
    print()


def demo_spatial_connectivity():
    """Demonstrate spatial connectivity patterns."""
    print("=" * 60)
    print("Demo 4: Spatial Connectivity Patterns")
    print("=" * 60)

    # Create 2D hillslope
    nx, ny = 50, 30
    state = ConnectivityState(shape=(ny, nx))

    # Create spatial fields
    x = np.linspace(0, 100, nx)
    y = np.linspace(0, 30, ny)
    X, Y = np.meshgrid(x, y)

    # Macroporosity: higher near stream (left side)
    macroporosity = 0.15 * np.exp(-X / 50)

    # rDUNE: increases away from stream
    rDUNE = 2.0 * (1 - np.exp(-X / 30))

    # Set E_crit based on structure and topography
    state.set_E_crit(
        E_crit=None,
        macroporosity=macroporosity,
        rDUNE=rDUNE,
        E_base=-1000.0,
        alpha_macro=500.0,
        beta_rDUNE=100.0
    )

    # Simulate rainfall event: increasing moisture
    theta_field = 0.15 + 0.20 * np.random.rand(ny, nx)

    # Calculate free energy (assuming HAND increases away from stream)
    HAND = 0.5 * X**0.5
    fe_calc = FreeEnergyCalculator(
        theta_r=0.05, theta_s=0.45, alpha=2.0, n=1.5,
        model='vanGenuchten'
    )
    E_free = np.array([[fe_calc.calculate_free_energy(theta_field[i, j], HAND[i, j])
                       for j in range(nx)] for i in range(ny)])

    # Update connectivity
    state.update(E_free, mode='sigmoid')

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # E_crit
    im0 = axes[0, 0].imshow(state.E_crit, aspect='auto', cmap='RdYlGn_r')
    axes[0, 0].set_title('Critical Energy E_crit [J/m³]')
    axes[0, 0].set_xlabel('Distance from Stream [m]')
    axes[0, 0].set_ylabel('Lateral Position [m]')
    plt.colorbar(im0, ax=axes[0, 0])

    # E_free
    im1 = axes[0, 1].imshow(E_free, aspect='auto', cmap='viridis')
    axes[0, 1].set_title('Free Energy E_free [J/m³]')
    axes[0, 1].set_xlabel('Distance from Stream [m]')
    axes[0, 1].set_ylabel('Lateral Position [m]')
    plt.colorbar(im1, ax=axes[0, 1])

    # Connectivity
    im2 = axes[1, 0].imshow(state.kappa, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    axes[1, 0].set_title('Connectivity κ [-]')
    axes[1, 0].set_xlabel('Distance from Stream [m]')
    axes[1, 0].set_ylabel('Lateral Position [m]')
    plt.colorbar(im2, ax=axes[1, 0])

    # Active network
    active_mask = state.get_active_mask(threshold=0.5)
    axes[1, 1].imshow(active_mask, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Active Network (κ > 0.5)\nActive Fraction: {state.get_active_fraction():.2%}')
    axes[1, 1].set_xlabel('Distance from Stream [m]')
    axes[1, 1].set_ylabel('Lateral Position [m]')

    plt.tight_layout()
    plt.savefig('/home/user/soilstocenergy/examples/spatial_connectivity.png', dpi=150, bbox_inches='tight')
    print("✓ Spatial connectivity plot saved to examples/spatial_connectivity.png")

    # Print statistics
    pattern = state.get_connectivity_pattern()
    print(f"\nConnectivity Statistics:")
    print(f"  Mean connectivity: {pattern['mean']:.3f}")
    print(f"  Active fraction (κ > 0.5): {pattern['active_fraction']:.1%}")
    print(f"  Fully active (κ > 0.9): {pattern['fully_active']:.1%}")
    print(f"  Disconnected (κ < 0.1): {pattern['disconnected']:.1%}")
    print()


def demo_hysteresis():
    """Demonstrate hysteresis in connectivity."""
    print("=" * 60)
    print("Demo 5: Hysteresis in Wetting-Drying Cycles")
    print("=" * 60)

    # Create hysteresis connectivity
    hyst = HysteresisConnectivity(
        E_crit_wet=-50.0,   # Higher threshold for wetting
        E_crit_dry=-150.0,  # Lower threshold for drying
    )

    # Simulate wetting-drying cycle
    # Wetting phase
    E_wetting = np.linspace(-300, 100, 100)
    kappa_wetting = []
    hyst.previous_E_free = -300.0

    for E in E_wetting:
        k = hyst.calculate_connectivity(E, mode='sigmoid')
        kappa_wetting.append(k)

    # Drying phase
    E_drying = np.linspace(100, -300, 100)
    kappa_drying = []

    for E in E_drying:
        k = hyst.calculate_connectivity(E, mode='sigmoid')
        kappa_drying.append(k)

    # Plot hysteresis loop
    plt.figure(figsize=(10, 6))
    plt.plot(E_wetting, kappa_wetting, 'b-', linewidth=2, label='Wetting', marker='>')
    plt.plot(E_drying, kappa_drying, 'r-', linewidth=2, label='Drying', marker='<')
    plt.axvline(-50, color='b', linestyle='--', alpha=0.5, label='E_crit_wet')
    plt.axvline(-150, color='r', linestyle='--', alpha=0.5, label='E_crit_dry')
    plt.xlabel('Free Energy E_free [J/m³]')
    plt.ylabel('Connectivity κ [-]')
    plt.title(f'Hysteresis Loop (ΔE_crit = {hyst.get_hysteresis_width():.0f} J/m³)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/user/soilstocenergy/examples/hysteresis.png', dpi=150, bbox_inches='tight')
    print("✓ Hysteresis plot saved to examples/hysteresis.png")
    print()


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("SoilStocEnergy Phase 1-3 Demonstration")
    print("Thermodynamics and Connectivity Framework")
    print("=" * 60 + "\n")

    # Run demonstrations
    demo_free_energy()
    demo_rdune()
    demo_connectivity_modes()
    demo_spatial_connectivity()
    demo_hysteresis()

    print("=" * 60)
    print("All demonstrations completed successfully!")
    print("Plots saved to examples/ directory")
    print("=" * 60)


if __name__ == '__main__':
    main()
