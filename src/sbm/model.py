from typing import Dict, Optional
import numpy as np
from scipy.sparse import csr_array

from sbm.block_data import BlockData
from sbm.likelihood import (
    LikelihoodCalculator,
    LikelihoodType,
)
from sbm.block_change_proposers import NodeSwapProposer
from sbm.node_mover import NodeMover
from sbm.mcmc import MCMCAlgorithm

from sbm.io import SBMFit
from sbm.utils.logger import CSVLogger

class SBMModel:
    def __init__(self,
                initial_blocks: BlockData,
                rng: np.random.Generator,
                likelihood_type: LikelihoodType = "bernoulli",
                log: bool = True
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

        self.mcmc_algorithm = MCMCAlgorithm(
            block_data = self.block_data,
            likelihood_calculator = self.likelihood_calculator,
            change_proposer = {
                "swap": NodeSwapProposer(
                            block_data=self.block_data,
                            rng=self.rng,
                            use_numpy=True
                        )
                },
            rng = self.rng,
            log=log
        )

    def fit(self,
            max_num_iterations: int,
            min_block_size: int,
            initial_temperature: float,
            cooling_rate: float,
            max_blocks: Optional[int] = None,
            logger: Optional[CSVLogger] = None,
            patience: Optional[int] = None,
            ):

        self.mcmc_algorithm.fit(
            max_num_iterations=max_num_iterations,
            min_block_size=min_block_size,
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            max_blocks=max_blocks,
            logger=logger,
        )

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
            ll = self.mcmc_algorithm.best_ll
        else:
            blocks = self.block_data.blocks
            block_sizes = list(self.block_data.block_sizes.values())
            block_conn = self.block_data.block_connectivity
            ll = self.mcmc_algorithm.current_ll

        return SBMFit(
            block_sizes=block_sizes,
            block_conn=csr_array(block_conn),
            directed_graph=self.block_data.graph_data.directed,
            neg_loglike=ll,
            metadata=metadata
        )