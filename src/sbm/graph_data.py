from scipy.sparse import csr_matrix

class GraphData:
    def __init__(self, adjacency_matrix: csr_matrix, directed=False):
        if not isinstance(adjacency_matrix, csr_matrix):
            raise ValueError("Adjacency matrix must be a scipy.sparse.csr_matrix")

        self.adjacency = adjacency_matrix
        
        if adjacency_matrix.shape is not None:
            self.num_nodes = adjacency_matrix.shape[0]
        else:
            raise ValueError("Adjacency has no shape attribute")
        
        self.total_edges = self.adjacency.sum()

        if directed:
            self.total_edges = self.adjacency.sum()
        else:
            self.total_edges = self.adjacency.sum() / 2  # For undirected graphs
        
        def __len__(self):
            return self.num_nodes