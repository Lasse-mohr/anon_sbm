from typing import Optional, Tuple, Dict
import numpy as np

#from src.sbm.graph_data import GraphData
from src.sbm.block_data import BlockData
from src.sbm.likelihood import LikelihoodCalculator
from src.sbm.move import MoveProposer, MoveExecutor

class MCMCAlgorithm:
    def __init__(self,
                 block_data: BlockData,
                 likelihood_calculator: LikelihoodCalculator,
                 move_proposer: MoveProposer,
                 move_executor: MoveExecutor,
                 seed: Optional[int] = None):

        self.acceptance_counts = {
            'move_node': {'proposed': 0, 'accepted': 0},
            'split_block': {'proposed': 0, 'accepted': 0},
            'merge_blocks': {'proposed': 0, 'accepted': 0}
            }
        self.move_probabilities = {
            'move_node': 1/3,
            'split_block': 1/3,
            'merge_blocks': 1/3
            }
        self.block_data = block_data
        self.likelihood_calculator = likelihood_calculator
        self.move_proposer = move_proposer
        self.move_executor = move_executor
        self.rng = np.random.default_rng(seed)
        self.temperature = None
        self.current_ll = self.likelihood_calculator.compute_likelihood()
        # Initialize acceptance counts and move probabilities

    def fit(self, num_iterations: int, min_block_size: int, initial_temperature: float, cooling_rate: float,
                target_acceptance_rate: float = 0.25, max_blocks: Optional[int] = None) -> None:
        """
        Run the adaptive MCMC algorithm to fit the SBM to the network data.

        :param num_iterations: Total number of MCMC iterations to run.
        :param min_block_size: Minimum allowed size for any block.
        :param initial_temperature: Starting temperature for simulated annealing.
        :param cooling_rate: Rate at which temperature decreases.
        :param target_acceptance_rate: Desired acceptance rate for adaptive adjustments (default 25%).
        :param max_blocks: Optional maximum number of blocks allowed.
        """
        temperature = initial_temperature
        current_ll = self.likelihood_calculator.compute_likelihood()

        for iteration in range(1, num_iterations + 1):
            move_type = self._select_move_type()
            delta_ll, move_accepted = self._attempt_move(move_type, min_block_size, temperature, max_blocks)
            # Update acceptance counts
            self.acceptance_counts[move_type]['proposed'] += 1
            if move_accepted:
                self.acceptance_counts[move_type]['accepted'] += 1
                current_ll += delta_ll

            # Update temperature
            temperature = self._update_temperature(temperature, cooling_rate)

            # Log and adjust proposal probabilities every 100 iterations
            if iteration % 100 == 0:
                acceptance_rates = {
                    move: self.acceptance_counts[move]['accepted'] / max(self.acceptance_counts[move]['proposed'], 1)
                    for move in self.move_probabilities
                }
                self._update_move_probabilities(acceptance_rates, target_acceptance_rate)
                self._log_iteration(iteration, current_ll, temperature, acceptance_rates)

    def _select_move_type(self) -> str:
        """
        Select a move type based on the current proposal probabilities.

        :return: The selected move type.
        """
        move_type = self.rng.choice(list(self.move_probabilities.keys()), p=list(self.move_probabilities.values()))
        return move_type

    def _attempt_move(self, move_type: str, min_block_size: int, temperature: float,
                      max_blocks: Optional[int]) -> Tuple[float, bool]:
        """
        Attempt a move of the specified type.

        :param move_type: The type of move to attempt ('move_node', 'split_block', 'merge_blocks').
        :param min_block_size: Minimum allowed size for any block.
        :param temperature: Current temperature for simulated annealing.
        :param max_blocks: Optional maximum number of blocks allowed.
        :return: Tuple of (delta_ll, move_accepted)
        """
        delta_ll, move_accepted = 0.0, False

        if move_type == 'move_node':
            # Propose moving a node between blocks
            proposal = self.move_proposer.propose_move_node(min_block_size)
            if proposal:
                node, source_block, target_block = proposal
                # Compute change in log-likelihood and accept/reject move
                delta_ll = self.likelihood_calculator._compute_delta_ll_move_node(
                    node=node,
                    source_block=source_block,
                    target_block=target_block
                    )

                move_accepted = self._accept_move(
                    delta_ll=delta_ll,
                    temperature=temperature)

                if move_accepted:
                    self.move_executor.move_node(
                        node=node,
                        source_block=source_block,
                        target_block=target_block
                        )

        elif move_type == 'split_block':
            # propose splitting a block
            proposal = self.move_proposer.propose_split_block(min_block_size=min_block_size)
            if proposal and (max_blocks is None or len(self.block_data.block_members) < max_blocks):
                block_to_split, nodes_a, nodes_b = proposal

                # Compute change in log-likelihood and accept/reject move
                delta_ll = self.likelihood_calculator.compute_delta_ll_split_block(
                    block_to_split=block_to_split,
                    nodes_a=nodes_a,
                    nodes_b=nodes_b
                    )
                move_accepted = self._accept_move(delta_ll, temperature)
                if move_accepted:
                    self.move_executor.split_block(
                        block_to_split=block_to_split,
                        nodes_a=nodes_a,
                        nodes_b=nodes_b
                        )

        elif move_type == 'merge_blocks':
            # Propose merging two blocks
            proposal = self.move_proposer.propose_merge_blocks()
            if proposal and len(self.block_data.block_members) > 1:
                block_a, block_b = proposal
                # Compute change in log-likelihood and accept/reject move
                delta_ll = self.likelihood_calculator._compute_delta_ll_merge_blocks(
                    block_a=block_a,
                    block_b=block_b
                    )
                move_accepted = self._accept_move(delta_ll, temperature)
                if move_accepted:
                    self.move_executor.merge_blocks(block_a, block_b)
        else:
            pass  # Unknown move type

        return delta_ll, move_accepted

    def _accept_move(self, delta_ll: float, temperature: float) -> bool:
        """
        Determine whether to accept a proposed move based on likelihood change and temperature.

        :param delta_ll: Change in log-likelihood resulting from the proposed move.
        :param temperature: Current temperature for simulated annealing.
        :return: True if move is accepted, False otherwise.
        """
        if delta_ll > 0:
            return True
        return self.rng.uniform() < np.exp(delta_ll / temperature)

    def _update_temperature(self, current_temperature: float, cooling_rate: float) -> float:
        """
        Update the temperature according to the cooling schedule.

        :param current_temperature: The current temperature.
        :param cooling_rate: The cooling rate.
        :return: The updated temperature.
        """
        return current_temperature * cooling_rate

    def _update_move_probabilities(self, acceptance_rates: Dict[str, float], target_acceptance_rate: float) -> None:
        """
        Adjust the move proposal probabilities based on acceptance rates.

        :param acceptance_rates: The acceptance rates for each move type.
        :param target_acceptance_rate: The desired overall acceptance rate.
        """
        adjustment = {move: acceptance_rates[move] / target_acceptance_rate if target_acceptance_rate > 0 else 1.0
                      for move in self.move_probabilities}
        total_adjustment = sum(adjustment.values())
        self.move_probabilities = {move: adjustment[move] / total_adjustment for move in self.move_probabilities}

    def _log_iteration(self, iteration: int, current_ll: float, temperature: float,
                       acceptance_rates: Dict[str, float]) -> None:
        """
        Log the current state of the algorithm for monitoring.

        :param iteration: The current iteration number.
        :param current_ll: The current log-likelihood.
        :param temperature: The current temperature.
        :param acceptance_rates: The acceptance rates for each move type.
        """
        print(f"Iteration {iteration}: LL={current_ll:.4f}, Temp={temperature:.4f}, "
              f"Acceptance Rates={acceptance_rates}")