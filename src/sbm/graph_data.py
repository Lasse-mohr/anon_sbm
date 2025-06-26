from scipy.sparse import csr_array
import networkx as nx

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

def gd_from_networkx(G: nx.Graph) -> GraphData:
    """
    Create a GraphData instance from a NetworkX graph.
    """
    if not hasattr(G, 'adjacency'):
        raise ValueError("The provided graph must have an adjacency matrix.")

    # for new version of networkx
    #adj = nx.to_scipy_sparse_matrix(G)

    # for old version of networkx
    adj = nx.to_scipy_sparse_matrix(G)
    adj = csr_array(adj)
    return GraphData(adj, directed=G.is_directed())