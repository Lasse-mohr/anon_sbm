import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from typing import Dict, Optional
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix


class SpectralPartitioner:
    def __init__(self, adjacency: csr_matrix, seed: Optional[int] = None):
        """
        Initialize the SpectralPartitioner.

        :param adjacency: The adjacency matrix of the network (sparse CSR matrix).
        :param seed: Random seed for reproducibility.
        """
        self.adjacency: csr_matrix = adjacency.tocsr()
        self.num_nodes: int = self.adjacency.shape[0]
        self.rng = np.random.default_rng(seed)
        self.degree_vector = np.array(self.adjacency.sum(axis=1)).flatten()
        self.laplacian = self._compute_normalized_laplacian()

    def _compute_normalized_laplacian(self) -> csr_matrix:
        """
        Compute the normalized Laplacian matrix of the graph.

        :return: The normalized Laplacian matrix (sparse CSR matrix).
        """
        # Avoid division by zero
        with np.errstate(divide='ignore'):
            d_inv_sqrt = np.power(self.degree_vector, -0.5)
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

        D_inv_sqrt = sp.diags(d_inv_sqrt)
        L = sp.eye(self.num_nodes) - D_inv_sqrt @ self.adjacency @ D_inv_sqrt
        return L

    def partition(self, num_blocks: int) -> Dict[int, int]:
        """
        Partition the nodes into blocks using spectral clustering.

        :param num_blocks: The desired number of blocks.
        :return: A dictionary mapping node indices to block indices.
        """
        # Compute the first (num_blocks) eigenvectors of the normalized Laplacian
        # Use 'SM' to find eigenvalues closest to zero
        eigenvalues, eigenvectors = eigsh(
            self.laplacian, k=num_blocks, which='SM', tol=1e-6, maxiter=5000
        )

        # Normalize rows to unit length to improve clustering
        embedding = eigenvectors
        row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1e-10  # Avoid division by zero
        embedding_normalized = embedding / row_norms

        # Use k-means clustering on the spectral embeddings
        kmeans = KMeans(n_clusters=num_blocks, random_state=self.rng.integers(1 << 32))
        labels = kmeans.fit_predict(embedding_normalized)

        # Map nodes to blocks
        blocks = {node: int(label) for node, label in enumerate(labels)}
        return blocks