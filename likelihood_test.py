import scipy.sparse as sp
import numpy as np
from src.sbm import StochasticBlockModel

# Example usage
if __name__ == "__main__":
    num_nodes = 1000
    block_size = 50
    # Generate a random sparse adjacency matrix
    adjacency = sp.random(num_nodes, num_nodes, density=0.01, format='csr')
    adjacency = adjacency + adjacency.T  # Make it symmetric
    adjacency.data = np.ones_like(adjacency.data)  # Unweighted graph

    # Create an initial uniform random partition
    initial_blocks = StochasticBlockModel.create_uniform_partition(num_nodes, block_size)

    # Initialize the SBM
    sbm = StochasticBlockModel(adjacency, initial_blocks)

    # Compute the initial likelihood
    initial_likelihood = sbm.compute_likelihood()
    print(f"Initial Likelihood: {initial_likelihood}")

    # Perform some partition manipulations
    sbm.split_block(0)
    sbm.merge_blocks(1, 2)
    sbm.move_node(10, 3)

    # Compute the new likelihood
    new_likelihood = sbm.compute_likelihood()
    print(f"New Likelihood: {new_likelihood}")

    # Check minimum block size
    min_size = sbm.min_block_size()
    print(f"Minimum Block Size: {min_size}")
