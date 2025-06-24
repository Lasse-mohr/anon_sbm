from scipy.sparse import csr_array

class GraphData:
    def __init__(self, adjacency_matrix: csr_array, directed=False):
        if not isinstance(adjacency_matrix, csr_array):
            raise ValueError("Adjacency matrix must be a scipy.sparse.csr_array")

        self.adjacency = adjacency_matrix.astype(int)
        self.directed: bool= directed
        self.num_nodes = self.adjacency.shape[0] # type: ignore
        
        if directed:
            self.total_edges = int(self.adjacency.sum())
        else:
            self.total_edges = int(self.adjacency.sum() / 2)  # For undirected graphs
        
        def __len__(self):
            return self.num_nodes