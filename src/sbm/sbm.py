from typing import Dict, Optional
from scipy.sparse import csr_matrix
from sbm.graph_data import GraphData
from sbm.block_data import BlockData
from sbm.likelihood import LikelihoodCalculator
from src.sbm.move_managers import MoveProposer, MoveExecutor
from sbm.util import set_random_seed
from sbm.mcmc import MCMCAlgorithm

class SBMModel:
    def __init__(self, adjacency_matrix: csr_matrix,
                 initial_blocks: Dict[int, int],
                 seed: int=1):

        self.block_data = BlockData(
            initial_blocks=initial_blocks,
            graph_data=GraphData(adjacency_matrix)
            )
        self.rng = set_random_seed(seed)

        self.likelihood_calculator = LikelihoodCalculator(
            block_data=self.block_data
            )
        self.move_proposer = MoveProposer(
            block_data=self.block_data,
            rng=self.rng
            )
        self.move_executor = MoveExecutor(
            block_data=self.block_data,
            )

        self.mcmc_algorithm = MCMCAlgorithm(
            block_data=self.block_data,
            likelihood_calculator=self.likelihood_calculator,
            move_proposer=self.move_proposer,
            move_executor=self.move_executor,
            seed=seed
        )

    def fit(self, num_iterations: int, min_block_size: int, initial_temperature: float, cooling_rate: float,
            target_acceptance_rate: float = 0.25, max_blocks: Optional[int] = None):

        self.mcmc_algorithm.fit(
            num_iterations=num_iterations,
            min_block_size=min_block_size,
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            target_acceptance_rate=target_acceptance_rate,
            max_blocks=max_blocks
        )

    def get_block_assignments(self) -> Dict[int, int]:
        return self.block_data.blocks
