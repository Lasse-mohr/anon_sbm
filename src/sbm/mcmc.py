""" 
Code for the MCMC algorithm used in the Stochastic Block Model (SBM) fitting.
This code implements the MCMC algorithm for both standard and private partitioning scenarios.
It includes the main MCMC algorithm class and a specialized class for private partitioning with differential privacy
"""
from typing import Optional, Tuple, Dict, Optional, List
from collections import deque
import numpy as np

#from line_profiler import profile
from numba import jit
from math import exp

#from src.sbm.graph_data import GraphData
from sbm.block_data import BlockData
from sbm.likelihood import LikelihoodCalculator
from sbm.block_change_proposers import ChangeProposer
from sbm.node_mover import NodeMover
from sbm.utils.logger import CSVLogger
from sbm.mcmc_diagnostics import OnlineDiagnostics

from sbm.block_change_proposers import ChangeProposer, ChangeProposerName

#### Aliases
ChangeProposerDict = Dict[ChangeProposerName, ChangeProposer] 
ChangeFreqDict = Dict[ChangeProposerName, float]

class MCMC:
    def __init__(self,
                 block_data: BlockData,
                 likelihood_calculator: LikelihoodCalculator,
                 change_proposer: ChangeProposerDict,
                 rng: np.random.Generator,
                 logger: Optional[CSVLogger] = None,
                 monitor: bool = True,
                 diag_lag: int =1000,
                 diag_checkpoints: int = 3000,
                 change_freq: Optional[ChangeFreqDict] = None,
                 ):

        self.block_data = block_data
        self.likelihood_calculator = likelihood_calculator
        self.change_proposers = change_proposer
        self.change_freq = change_freq
        self.node_mover = NodeMover(block_data=block_data)
        self.rng = rng
        self.current_nll = self.likelihood_calculator.nll
        self.logger = logger # True if logging is enabled, False otherwise.

        ### set up mcmc diagnostics (R̂ and ESS)
        # only for estimating convergence diagnostics for dp patitioning
        self._monitor = monitor
        if monitor:
            self._diag= OnlineDiagnostics(window=diag_lag)
            self._diag_checkpoints = diag_checkpoints
            self._off_diag = self._select_off_pairs(max_panel=diag_checkpoints)
        else:
            self._diag = None
            self._diag_checkpoints = 0
            self._off_diag = []

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
        acc_hist = deque(maxlen=1000)          # for accept‑rate window
        temperature = initial_temperature
        current_nll = self.likelihood_calculator.nll
        acceptance_rate = 0 # acceptance rate of moves between logging
        nll_list = [current_nll]

        # if patience None, set based on the graph size
        if patience is None:
            n_nodes = self.block_data.graph_data.num_nodes
            patience = min(int(0.1 * n_nodes*(n_nodes - 1) // 2), 10**5)

        if self.logger:
            self.logger.log(0, current_nll, acceptance_rate, temperature)

        n_steps_declined = 0
        for iteration in range(1, max_num_iterations + 1):
            move_type = self._select_move_type()

            delta_nll, move_accepted = self._attempt_move(
                move_type=move_type,
                min_block_size=min_block_size,
                temperature=temperature,
                max_blocks=max_blocks
                )
            acc_hist.append(move_accepted)

            # --- diagnostics update --------------------------------
            if self._monitor and self._diag is not None:
                diag_vec = self.block_data.diagonal_counts()
                off_vec  = self.block_data.counts_for_pairs(self._off_diag)
                self._diag.update(self.current_nll, diag_vec, off_vec)

            # update likelihood and best assignment so far
            if move_accepted :
                self.current_nll += delta_nll
                n_steps_declined = 0
                if self.logger:
                    acceptance_rate += 1

                if self.current_nll < self.best_nll:
                    self.best_nll = current_nll
                    self._best_block_assignment = self.block_data.blocks.copy()
                    self._best_block_conn = self.block_data.block_connectivity.copy()
            else:
                n_steps_declined += 1
            
            nll_list.append(self.current_nll)

            temperature = self._update_temperature(temperature, cooling_rate)

            # --- logging --------------------------------
            if (self.logger is not None and
                iteration % self.logger.log_every == 0
                ):
                if (self._monitor and
                    self._diag is not None and
                    iteration % self._diag_checkpoints == 0
                    ):
                    rhat, ess = self._diag.summary()
                else:
                    rhat, ess = np.nan, np.nan

                self.logger.log(
                    iteration          = iteration,
                    neg_loglike        = self.current_nll,
                    accept_rate_window = float(np.mean(acc_hist or [0])),
                    temperature        = 1.0,          # or self.T if annealing
                    rhat_max           = rhat,
                    ess_min            = ess,
                )
            
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

    # --------------------------------------------------------------
    def _select_off_pairs(self, max_panel: int):
        """ 
        Randomly sample off pairs to monitor for diagnostics. 

        First we select pairs that currently have at least one edge,
        then we add random pairs until we reach the desired size.
        """
        B = len(self.block_data.block_sizes)
        want = min(max_panel, 2 * B) # number of pairs to sample and track

        ### get all pairs that currently have >= 1 edge
        nz = [(r, s) for r in range(B) for s in range(r + 1, B)
              if self.block_data.block_connectivity[r, s] > 0]

        self.rng.shuffle(nz)
        panel = nz[:want]

        # 2. add random extras until size == want
        while len(panel) < want:
            r, s = self.rng.choice(range(B), 2)
            if r > s:
                r, s = s, r
            if (r, s) not in panel:
                panel.append((r, s))

        return panel

### --------------------------------------------------------------------------- 
#### MCMC Algorithm for Private Partitioning
### --------------------------------------------------------------------------- 
class PrivatePartitionMCMC(MCMC):
    """
    MCMC sampler that targets the node-level DP Boltzmann distribution
        P(z) ∝ exp(-ε · L(z) / (2Δ))
    where L is the negative log-likelihood and Δ is its global
    sensitivity (Δ = 1 for Bernoulli SBMs).
    """
    def __init__(self, *, epsilon: float, delta_ll_sensitivity=1.0, temperature=1, **kwargs):
        super().__init__(**kwargs)
        self.eps   = float(epsilon)
        self.delta = float(delta_ll_sensitivity)
        self.temperatur = temperature

    # --- single move ----------------------------------------------------
    def _accept_prob(self, delta_nll: float) -> float:
        """Metropolis ratio specialised to the exponential mechanism."""
        ratio = exp(- self.eps * delta_nll / (2.0 * self.delta))
        return min(1.0, ratio)

    def _attempt_move(self,
                      move_type: ChangeProposerName,
                      temperature: float,
                      max_blocks: Optional[int] = None,
                      min_block_size: Optional[int] = None,
                      ) -> Tuple[float, bool]:
        # invoke parent proposer / Δnll code
        delta_nll, accepted = super()._attempt_move(
            move_type=move_type,
            temperature=temperature,      # no annealing in a DP chain
            max_blocks=max_blocks,
            min_block_size=min_block_size
        )
        return delta_nll, accepted