"""
Percolation network analysis for dynamic connectivity.

This module implements percolation theory concepts for soil water systems:
- Network topology representation
- Cluster identification (union-find algorithm)
- Percolation threshold detection
- Scaling laws and critical exponents
- Backbone identification

Key concepts:
- Percolation threshold p_c: Critical occupation probability
- Universal scaling laws near critical point
- Spanning clusters and connectivity
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict


class UnionFind:
    """
    Union-Find (Disjoint Set Union) data structure for cluster identification.

    Efficient algorithm for tracking connected components in a network.
    Uses path compression and union by rank for near-constant time operations.
    """

    def __init__(self, n: int):
        """
        Initialize Union-Find structure.

        Parameters
        ----------
        n : int
            Number of elements (nodes)
        """
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n  # Size of each cluster

    def find(self, x: int) -> int:
        """
        Find root of element x with path compression.

        Parameters
        ----------
        x : int
            Element index

        Returns
        -------
        int
            Root of the cluster containing x
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """
        Unite clusters containing x and y.

        Parameters
        ----------
        x, y : int
            Element indices

        Returns
        -------
        bool
            True if union was performed, False if already in same cluster
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
        else:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
            self.rank[root_x] += 1

        return True

    def connected(self, x: int, y: int) -> bool:
        """
        Check if x and y are in the same cluster.

        Parameters
        ----------
        x, y : int
            Element indices

        Returns
        -------
        bool
            True if connected
        """
        return self.find(x) == self.find(y)

    def get_cluster_size(self, x: int) -> int:
        """
        Get size of cluster containing x.

        Parameters
        ----------
        x : int
            Element index

        Returns
        -------
        int
            Cluster size
        """
        return self.size[self.find(x)]

    def get_all_clusters(self) -> Dict[int, List[int]]:
        """
        Get all clusters as dictionary: root -> members.

        Returns
        -------
        dict
            Dictionary mapping cluster roots to member lists
        """
        clusters = defaultdict(list)
        for i in range(len(self.parent)):
            root = self.find(i)
            clusters[root].append(i)
        return dict(clusters)


class PercolationNetwork:
    """
    Percolation network for analyzing connectivity in soil systems.

    Represents soil elements as nodes and connections as bonds.
    Tracks active/inactive status and identifies percolating clusters.
    """

    def __init__(self, shape: Tuple[int, ...], connectivity: str = 'nearest'):
        """
        Initialize percolation network.

        Parameters
        ----------
        shape : tuple
            Shape of network (e.g., (nz,) for 1D, (ny, nx) for 2D)
        connectivity : str
            'nearest' for nearest-neighbor, 'extended' for extended neighbors
        """
        self.shape = shape
        self.n_elements = np.prod(shape)
        self.connectivity_type = connectivity
        self.dim = len(shape)

        # Network state
        self.active = np.zeros(shape, dtype=bool)
        self.bonds = self._create_bond_structure()

    def _create_bond_structure(self) -> List[Tuple[int, int]]:
        """
        Create bond structure based on connectivity type.

        Returns
        -------
        list
            List of (i, j) bond pairs (flat indices)
        """
        bonds = []

        if self.dim == 1:
            # 1D: vertical connections
            for i in range(self.shape[0] - 1):
                bonds.append((i, i + 1))

        elif self.dim == 2:
            # 2D: nearest neighbor (von Neumann)
            ny, nx = self.shape
            for i in range(ny):
                for j in range(nx):
                    idx = i * nx + j
                    # Right neighbor
                    if j < nx - 1:
                        bonds.append((idx, idx + 1))
                    # Bottom neighbor
                    if i < ny - 1:
                        bonds.append((idx, idx + nx))

        elif self.dim == 3:
            # 3D: nearest neighbor
            nz, ny, nx = self.shape
            for i in range(nz):
                for j in range(ny):
                    for k in range(nx):
                        idx = i * ny * nx + j * nx + k
                        # Right
                        if k < nx - 1:
                            bonds.append((idx, idx + 1))
                        # Back
                        if j < ny - 1:
                            bonds.append((idx, idx + nx))
                        # Down
                        if i < nz - 1:
                            bonds.append((idx, idx + ny * nx))

        return bonds

    def update_active_state(self, kappa: np.ndarray, threshold: float = 0.5):
        """
        Update active state based on connectivity field.

        Parameters
        ----------
        kappa : array
            Connectivity field κ(x) ∈ [0, 1]
        threshold : float
            Threshold for considering element active
        """
        if kappa.shape != self.shape:
            raise ValueError(f"kappa shape {kappa.shape} does not match network shape {self.shape}")

        self.active = kappa > threshold

    def get_active_bonds(self) -> List[Tuple[int, int]]:
        """
        Get list of active bonds (both endpoints active).

        Returns
        -------
        list
            List of active (i, j) bond pairs
        """
        active_flat = self.active.flatten()
        active_bonds = [(i, j) for i, j in self.bonds
                       if active_flat[i] and active_flat[j]]
        return active_bonds

    def identify_clusters(self) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        Identify connected clusters using Union-Find.

        Returns
        -------
        labels : array
            Cluster labels for each element (-1 for inactive)
        cluster_sizes : dict
            Dictionary mapping cluster_id -> size
        """
        uf = UnionFind(self.n_elements)
        active_flat = self.active.flatten()

        # Unite active bonds
        for i, j in self.get_active_bonds():
            uf.union(i, j)

        # Create labels
        labels = np.full(self.n_elements, -1, dtype=int)
        cluster_map = {}  # Map root to cluster_id
        cluster_id = 0

        for i in range(self.n_elements):
            if active_flat[i]:
                root = uf.find(i)
                if root not in cluster_map:
                    cluster_map[root] = cluster_id
                    cluster_id += 1
                labels[i] = cluster_map[root]

        labels = labels.reshape(self.shape)

        # Calculate cluster sizes
        cluster_sizes = {}
        for cid in range(cluster_id):
            cluster_sizes[cid] = np.sum(labels == cid)

        return labels, cluster_sizes

    def check_percolation(self, direction: Optional[str] = None) -> bool:
        """
        Check if network has spanning cluster (percolation).

        Parameters
        ----------
        direction : str, optional
            For 1D: None (end-to-end)
            For 2D/3D: 'vertical', 'horizontal', 'any'

        Returns
        -------
        bool
            True if percolating cluster exists
        """
        if not np.any(self.active):
            return False

        labels, _ = self.identify_clusters()

        if self.dim == 1:
            # Check if top and bottom are connected
            if self.active[0] and self.active[-1]:
                return labels[0] == labels[-1]
            return False

        elif self.dim == 2:
            ny, nx = self.shape
            if direction == 'vertical' or direction is None:
                # Check top-bottom spanning
                top_labels = set(labels[0, :][self.active[0, :]])
                bottom_labels = set(labels[-1, :][self.active[-1, :]])
                if top_labels & bottom_labels:  # Intersection
                    return True

            if direction == 'horizontal' or direction is None:
                # Check left-right spanning
                left_labels = set(labels[:, 0][self.active[:, 0]])
                right_labels = set(labels[:, -1][self.active[:, -1]])
                if left_labels & right_labels:
                    return True

            return False

        elif self.dim == 3:
            nz, ny, nx = self.shape
            if direction == 'vertical' or direction is None:
                # Check top-bottom in z
                top_labels = set(labels[0, :, :].flatten()[self.active[0, :, :].flatten()])
                bottom_labels = set(labels[-1, :, :].flatten()[self.active[-1, :, :].flatten()])
                if top_labels & bottom_labels:
                    return True

            return False

        return False

    def get_largest_cluster(self) -> Tuple[int, int]:
        """
        Get size and ID of largest cluster.

        Returns
        -------
        cluster_id : int
            ID of largest cluster (-1 if none)
        size : int
            Size of largest cluster
        """
        if not np.any(self.active):
            return -1, 0

        labels, cluster_sizes = self.identify_clusters()

        if not cluster_sizes:
            return -1, 0

        largest_id = max(cluster_sizes, key=cluster_sizes.get)
        largest_size = cluster_sizes[largest_id]

        return largest_id, largest_size

    def calculate_active_fraction(self) -> float:
        """
        Calculate fraction of active sites (occupation probability).

        Returns
        -------
        float
            p = N_active / N_total
        """
        return np.mean(self.active)

    def get_percolation_strength(self) -> float:
        """
        Calculate percolation strength (fraction in spanning cluster).

        Returns
        -------
        float
            P = N_spanning / N_total
        """
        if not self.check_percolation():
            return 0.0

        labels, cluster_sizes = self.identify_clusters()

        # Find spanning cluster(s)
        spanning_size = 0
        if self.dim == 1:
            if self.active[0] and self.active[-1] and labels[0] == labels[-1]:
                spanning_size = cluster_sizes[labels[0]]

        elif self.dim == 2:
            ny, nx = self.shape
            top_labels = set(labels[0, :][self.active[0, :]])
            bottom_labels = set(labels[-1, :][self.active[-1, :]])
            spanning_labels = top_labels & bottom_labels

            for cluster_id in spanning_labels:
                spanning_size += cluster_sizes[cluster_id]

        return spanning_size / self.n_elements


class ScalingLaws:
    """
    Calculate percolation scaling laws and critical exponents.

    Near the percolation threshold p_c, observables follow power laws:
    - Correlation length: ξ ∝ |p - p_c|^(-ν)
    - Cluster size: S ∝ |p - p_c|^(-γ)
    - Percolation strength: P ∝ (p - p_c)^β
    - Conductivity: K_eff ∝ (p - p_c)^μ
    """

    # Theoretical critical exponents (lattice-dependent)
    EXPONENTS_2D = {
        'nu': 4/3,      # Correlation length
        'gamma': 43/18, # Cluster size
        'beta': 5/36,   # Percolation strength
        'mu': 1.3,      # Conductivity (approximate)
    }

    EXPONENTS_3D = {
        'nu': 0.88,
        'gamma': 1.8,
        'beta': 0.41,
        'mu': 2.0,
    }

    # Percolation thresholds
    P_C = {
        '1d': 1.0,      # 1D always percolates if fully connected
        '2d_square': 0.5927,  # 2D square lattice (site percolation)
        '3d_cubic': 0.3116,   # 3D cubic lattice (site percolation)
    }

    @staticmethod
    def get_theoretical_exponents(dim: int) -> Dict[str, float]:
        """
        Get theoretical critical exponents for dimension.

        Parameters
        ----------
        dim : int
            Network dimension

        Returns
        -------
        dict
            Dictionary of critical exponents
        """
        if dim == 2:
            return ScalingLaws.EXPONENTS_2D.copy()
        elif dim == 3:
            return ScalingLaws.EXPONENTS_3D.copy()
        else:
            return {}

    @staticmethod
    def estimate_percolation_threshold(
        p_values: np.ndarray,
        percolating: np.ndarray
    ) -> float:
        """
        Estimate percolation threshold from simulations.

        Parameters
        ----------
        p_values : array
            Occupation probabilities tested
        percolating : array
            Boolean array indicating if percolation occurred

        Returns
        -------
        float
            Estimated p_c
        """
        if not np.any(percolating):
            return np.nan

        # Find transition point
        first_percolating = np.where(percolating)[0]
        if len(first_percolating) == 0:
            return np.nan

        idx = first_percolating[0]
        if idx == 0:
            return p_values[0]

        # Linear interpolation between last non-percolating and first percolating
        return (p_values[idx - 1] + p_values[idx]) / 2

    @staticmethod
    def fit_power_law(
        x: np.ndarray,
        y: np.ndarray,
        x_critical: float
    ) -> Tuple[float, float]:
        """
        Fit power law: y ∝ |x - x_c|^α

        Parameters
        ----------
        x : array
            Independent variable (e.g., p)
        y : array
            Dependent variable (e.g., cluster size)
        x_critical : float
            Critical point x_c

        Returns
        -------
        exponent : float
            Fitted exponent α
        prefactor : float
            Fitted prefactor A
        """
        # Remove critical point and zeros
        mask = (np.abs(x - x_critical) > 1e-6) & (y > 0)
        x_fit = np.abs(x[mask] - x_critical)
        y_fit = y[mask]

        if len(x_fit) < 2:
            return np.nan, np.nan

        # Log-log fit
        log_x = np.log(x_fit)
        log_y = np.log(y_fit)

        # Linear regression in log space
        coeffs = np.polyfit(log_x, log_y, 1)
        exponent = coeffs[0]
        prefactor = np.exp(coeffs[1])

        return exponent, prefactor


def calculate_correlation_length(network: PercolationNetwork) -> float:
    """
    Estimate correlation length from cluster size distribution.

    ξ ≈ sqrt(⟨s²⟩ / ⟨s⟩)

    Parameters
    ----------
    network : PercolationNetwork
        Network with active state set

    Returns
    -------
    float
        Correlation length
    """
    labels, cluster_sizes = network.identify_clusters()

    if not cluster_sizes:
        return 0.0

    sizes = np.array(list(cluster_sizes.values()))
    mean_s = np.mean(sizes)
    mean_s2 = np.mean(sizes**2)

    if mean_s > 0:
        xi = np.sqrt(mean_s2 / mean_s)
    else:
        xi = 0.0

    return xi


def calculate_mean_cluster_size(network: PercolationNetwork) -> float:
    """
    Calculate mean cluster size (excluding spanning cluster).

    S = ⟨s²⟩ / ⟨s⟩ for finite clusters

    Parameters
    ----------
    network : PercolationNetwork
        Network with active state set

    Returns
    -------
    float
        Mean cluster size
    """
    labels, cluster_sizes = network.identify_clusters()

    if not cluster_sizes:
        return 0.0

    # Exclude very large clusters (likely spanning)
    max_size = 0.1 * network.n_elements  # Threshold
    finite_sizes = [s for s in cluster_sizes.values() if s < max_size]

    if not finite_sizes:
        return 0.0

    sizes = np.array(finite_sizes)
    return np.mean(sizes**2) / np.mean(sizes)


def identify_backbone(network: PercolationNetwork) -> np.ndarray:
    """
    Identify backbone: bonds that carry flow in spanning cluster.

    The backbone is the subset of the spanning cluster through which
    flow actually occurs (removing dead-ends).

    Parameters
    ----------
    network : PercolationNetwork
        Network with active state set

    Returns
    -------
    array
        Boolean array indicating backbone elements

    Notes
    -----
    This is a simplified implementation. Full backbone identification
    requires solving flow equations or using burning algorithm.
    """
    if not network.check_percolation():
        return np.zeros(network.shape, dtype=bool)

    labels, cluster_sizes = network.identify_clusters()

    # Find spanning cluster
    if network.dim == 1:
        if network.active[0] and network.active[-1]:
            spanning_label = labels[0]
        else:
            return np.zeros(network.shape, dtype=bool)
    else:
        # For 2D/3D, find largest cluster as approximation
        largest_id, _ = network.get_largest_cluster()
        spanning_label = largest_id

    # All elements in spanning cluster (simplified backbone)
    backbone = (labels == spanning_label)

    return backbone
