from typing import Optional, Tuple, Dict, Optional, List
import numpy as np

#from line_profiler import profile
from numba import jit

#from src.sbm.graph_data import GraphData
from sbm.block_data import BlockData
from sbm.likelihood import LikelihoodCalculator
from sbm.block_change_proposers import ChangeProposer
from sbm.node_mover import NodeMover
from sbm.utils.logger import CSVLogger

from sbm.block_change_proposers import ChangeProposer, ChangeProposerName

#### Aliases
ChangeProposerDict = Dict[ChangeProposerName, ChangeProposer] 
ChangeFreqDict = Dict[ChangeProposerName, float]

class MCMCAlgorithm:
    def __init__(self,
                 block_data: BlockData,
                 likelihood_calculator: LikelihoodCalculator,
                 change_proposer: ChangeProposerDict,
                 rng: np.random.Generator,
                 log: bool = True,
                 change_freq: Optional[ChangeFreqDict] = None,
                 ):

        self.block_data = block_data
        self.likelihood_calculator = likelihood_calculator
        self.change_proposers = change_proposer
        self.change_freq = change_freq
        self.node_mover = NodeMover(block_data=block_data)
        self.rng = rng
        self.current_nll = self.likelihood_calculator.nll
        self.log = log # True if logging is enabled, False otherwise.

        # store the best block assignment and likelihood
        self._best_block_assignment = block_data.blocks.copy()
        self._best_block_conn = block_data.block_connectivity.copy()
        self.best_nll = self.likelihood_calculator.nll

    def fit(self,
            max_num_iterations: int,
            initial_temperature: float = 1,
            cooling_rate: float = 0.99,
            min_block_size: Optional[int] = None,
            max_blocks: Optional[int] = None,
            logger: Optional[CSVLogger] = None,
            patience: Optional[int] = None,
        ) -> List[float]:
        """
        Run the adaptive MCMC algorithm to fit the SBM to the network data.

        :param max_num_iterations: max number of MCMC iterations to run.
        :param min_block_size: Minimum allowed size for any block.
        :param initial_temperature: Starting temperature for simulated annealing.
        :param cooling_rate: Rate at which temperature decreases.
        :param target_acceptance_rate: Desired acceptance rate for adaptive adjustments (default 25%).
        :param max_blocks: Optional maximum number of blocks allowed.
        """
        temperature = initial_temperature
        current_nll = self.likelihood_calculator.nll
        acceptance_rate = 0 # acceptance rate of moves between logging
        nll_list = [current_nll]

        # if patience None, set based on the graph size
        if patience is None:
            n_nodes = self.block_data.graph_data.num_nodes
            patience = min(int(0.1 * n_nodes*(n_nodes - 1) // 2), 10**5)

        if logger:
            logger.log(0, current_nll, acceptance_rate, temperature)

        n_steps_declined = 0
        for iteration in range(1, max_num_iterations + 1):
            move_type = self._select_move_type()

            delta_nll, move_accepted = self._attempt_move(
                move_type=move_type,
                min_block_size=min_block_size,
                temperature=temperature,
                max_blocks=max_blocks
                )

            # update likelihood and best assignment so far
            if move_accepted :
                self.current_nll += delta_nll
                n_steps_declined = 0
                if logger:
                    acceptance_rate += 1

                if self.current_nll < self.best_nll:
                    self.best_nll = current_nll
                    self._best_block_assignment = self.block_data.blocks.copy()
                    self._best_block_conn = self.block_data.block_connectivity.copy()
            else:
                n_steps_declined += 1
            
            nll_list.append(self.current_nll)

            temperature = self._update_temperature(temperature, cooling_rate)

            if logger and iteration % logger.log_every == 0:
                acceptance_rate = acceptance_rate / logger.log_every
                logger.log(iteration, self.current_nll, acceptance_rate, temperature)
                acceptance_rate = 0
            
            if patience is not None and n_steps_declined >= patience:
                print(f"Stopping early after {iteration} iterations due to patience limit.")
                break

        return nll_list 

    def _select_move_type(self) -> ChangeProposerName:
        """
        Select a move type based on the current proposal probabilities.

        :return: The selected move type.
        """
        if self.change_freq is None:
            return "uniform_swap"
        else:
            # Select a move type based on the defined probabilities
            move_type = self.rng.choice(
                tuple(self.change_freq.keys()),
                p=tuple(self.change_freq.values())
            )
        
        return move_type # type: ignore

    def _attempt_move(self,
                      move_type: ChangeProposerName,
                      temperature: float,
                      max_blocks: Optional[int] = None,
                      min_block_size: Optional[int] = None,
        ) -> Tuple[float, bool]:
        """
        Attempt a move of the specified type.

        :param move_type: The type of move to attempt ('swap').
        :param min_block_size: Minimum allowed size for any block.
        :param temperature: Current temperature for simulated annealing.
        :param max_blocks: Optional maximum number of blocks allowed.
        :return: Tuple of (delta_nll, move_accepted)
        """
        delta_nll, move_accepted = 0.0, False

        proposed_change, proposed_delta_e, proposed_delta_n = \
            self.change_proposers[move_type].propose_change()

        # Compute change in log-likelihood and accept/reject move
        delta_nll = self.likelihood_calculator.compute_delta_nll(
            delta_e=proposed_delta_e,
            delta_n=proposed_delta_n
            )

        move_accepted = self._accept_move(delta_nll, temperature)
        if move_accepted:
            self.node_mover.perform_change(proposed_change, proposed_delta_e)
    
        return delta_nll, move_accepted

    def _accept_move(self, delta_nll: float, temperature: float, eps:float=1e-6) -> bool:
        """
        Determine whether to accept a proposed move based on likelihood change and temperature.

        :param delta_nll: Change in negative log-likelihood resulting from the proposed move.
        :param temperature: Current temperature for simulated annealing.
        :return: True if move is accepted, False otherwise.
        """
        if delta_nll < 0:
            return True

        temperature = max(temperature, eps)  # Avoid division by zero
        z = min(delta_nll / temperature, 700) # clip to avoid overflow in exp

        return self.rng.uniform() > np.exp(z)

    def _update_temperature(self, current_temperature: float, cooling_rate: float) -> float:
        """
        Update the temperature according to the cooling schedule.

        :param current_temperature: The current temperature.
        :param cooling_rate: The cooling rate.
        :return: The updated temperature.
        """
        return current_temperature * cooling_rate