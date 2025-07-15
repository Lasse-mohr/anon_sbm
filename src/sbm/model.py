from typing import Dict, Optional
import numpy as np
from scipy.sparse import csr_array

from sbm.block_data import BlockData
from sbm.likelihood import (
    LikelihoodCalculator,
    LikelihoodType,
)
from sbm.block_change_proposers import (
    NodeSwapProposer,
    EdgeBasedSwapProposer,
    TriadicSwapProposer,
    CrossTriangleSwapProposer,
    TwinLeafSwapProposer
)

from sbm.node_mover import NodeMover
from sbm.mcmc import MCMC, PrivatePartitionMCMC

from sbm.io import SBMFit
from sbm.utils.logger import CSVLogger

class SBMModel:
    def __init__(self,
                initial_blocks: BlockData,
                rng: np.random.Generator,
                likelihood_type: LikelihoodType = "bernoulli",
                logger: Optional[CSVLogger] = None,
                change_freq = { # probabilities of trying each move type
                    "edge_based_swap": 1.0,
                },
                private_sbm: bool = False, # whether to use the private partitioning MCMC
                eps: float = 1.0, # privacy budget for private partitioning
        ):

        self._best_block_assignment = None
        self._best_block_conn = None

        self.block_data = initial_blocks

        self.rng = rng

        self.likelihood_calculator = LikelihoodCalculator(
            block_data=self.block_data,
            likelihood_type=likelihood_type
            )
        self.move_executor = NodeMover(
            block_data=self.block_data,
            )

        change_proposer = {
            "uniform_swap": NodeSwapProposer(
                        block_data=self.block_data,
                        rng=self.rng,
                        use_numpy=True,
                    ),
            "edge_based_swap": EdgeBasedSwapProposer(
                        block_data=self.block_data,
                        rng=self.rng,
                        use_numpy=False,
                    ),
            "triadic_swap": TriadicSwapProposer(
                        block_data=self.block_data,
                        rng=self.rng,
                        use_numpy=True,
                    ),
            "twin_leaf": TwinLeafSwapProposer(
                        block_data=self.block_data,
                        rng=self.rng,
                        use_numpy=True,
                    ),
            "cross_triangle": CrossTriangleSwapProposer(
                        block_data=self.block_data,
                        rng=self.rng,
                        use_numpy=True,
                    ),
        }

        if private_sbm:
            self.mcmc_algorithm = PrivatePartitionMCMC(
                block_data=self.block_data,
                likelihood_calculator=self.likelihood_calculator,
                change_proposer=change_proposer, # type: ignore
                rng=self.rng,
                logger=logger,
                epsilon=eps,
            )
        else:
            self.mcmc_algorithm = MCMC(
                block_data = self.block_data,
                likelihood_calculator = self.likelihood_calculator,
                change_proposer = change_proposer, # type: ignore
                change_freq = change_freq, # type: ignore
                rng = self.rng,
                logger=logger
            )

    def fit(self,
        min_block_size: int,
        cooling_rate: float=1-1e-4,
        max_blocks: Optional[int] = None,
        patience: Optional[int] = None,
        return_nll: bool = False,
        max_num_iterations: int=int(10**6),
        initial_temperature: float=1.0,
        ):

        nll = self.mcmc_algorithm.fit(
            max_num_iterations=max_num_iterations,
            min_block_size=min_block_size,
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            max_blocks=max_blocks,
            patience=patience,
        )

        if return_nll:
            return nll

    def get_block_assignments(self, best:bool=True) -> Dict[int, int]:
        if best:
            return self.mcmc_algorithm._best_block_assignment
        else:
            return self.block_data.blocks
    
    def to_sbmfit(self, metadata: Optional[Dict] = None, best:bool=True) -> SBMFit:
        """ 
        Convert the fitted SBM model to an SBMFit object for serialization. 
        """
        if metadata is None:
            metadata = {}
        
        if best:
            blocks = self.mcmc_algorithm._best_block_assignment
            block_sizes = np.unique(list(blocks.values()), return_counts=True)[1].tolist()
            block_conn = self.mcmc_algorithm._best_block_conn
            nll = self.mcmc_algorithm.best_nll
        else:
            blocks = self.block_data.blocks
            block_sizes = list(self.block_data.block_sizes.values())
            block_conn = self.block_data.block_connectivity
            nll = self.mcmc_algorithm.current_nll

        return SBMFit(
            block_sizes=block_sizes,
            block_conn=csr_array(block_conn),
            directed_graph=self.block_data.graph_data.directed,
            neg_loglike=nll,
            metadata=metadata
        )