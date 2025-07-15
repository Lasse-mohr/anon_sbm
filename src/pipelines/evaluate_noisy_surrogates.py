"""
    Generate surrogate differentially private SBM graphs and evaluate them against empirical data.
"""
import argparse
import yaml
import csv
from pathlib import Path
import numpy as np

from itertools import product
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from metrics import REGISTRY

from sbm.io import SBMWriter
from sbm.sampling import sample_sbm_graph_from_fit

from sbm.io import GraphLoader

from sbm.utils.pipeline_utils import (
    sbmfit_folderpath,
    surrogate_statistics_filename,
    FitConfig,
    EvalConfig,
)
from sbm.noisy_fit import create_sbm_noise

#######################
### Configuration Types 
#######################

def _generate_and_evaluate(
        sbm_config, # fitting configs of the SBM
        ds, # dataset specifications
        eval_config, # evaluation configurations
        rng: np.random.Generator,
        eps:float, # privacy level epsilon
        delta_sum:float, # sum of privacy level delta and alpha
    ):

    # split the privacy level delta (probability of failure)
    delta, alpha = delta_sum/2, delta_sum/2

    # load fitted model
    fit_folder_path = sbmfit_folderpath(
        base_dir=Path("results/sbm_fits"),
        sbm_config=sbm_config,# type: ignore
        data_spec=ds,
    )
    # load the sbm fit
    sbm_fit = SBMWriter.load(fit_folder_path)

    noise_factory = create_sbm_noise(
        sbm=sbm_fit,
        eps=eps,
        delta=delta,
        alpha=alpha,
        noise_type="heterogeneous_gaussian"
    )


    # check if metrics have been cached earlier
    out = surrogate_statistics_filename(
        base_dir=Path("results/surrogate_statistics/dp"),
        eval_configs=eval_config,
        sbm_config=sbm_config,# type: ignore
        data_spec=ds,
    )

    out.parent.mkdir(exist_ok=True)
    if out.exists() and not eval_config["overwrite"]:
        return

    # Generate surrogates and campare metrics
    results = []
    for i in tqdm(range(eval_config["n_surrogates"])):
        # sample an SBM graph from the sbm fit
        emp = sample_sbm_graph_from_fit(sbm_fit, rng)

        # sample differentially private sbm-fit
        lasso_noisy_fit = noise_factory.sample_sbm_fit(rng, post='lasso') # type: ignore

        # sample graph from the noisy sbm fit
        private_surr = sample_sbm_graph_from_fit(lasso_noisy_fit, rng)
        private_surr = private_surr.adjacency

        row = {"dataset": ds["name"], "surrogate": f'surr_{i}'}
        for m in eval_config["metrics"]:
            row[m] = REGISTRY[m](emp, private_surr)

        results.append(row)

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)

############################################
### main function
############################################

def main(fit_config: str, eval_config: str, parallel:bool): # type: ignore

    fit_config: FitConfig = yaml.safe_load(Path(fit_config).read_text())
    rng = np.random.default_rng(fit_config["seed"])
    data_config = fit_config["datasets"] # type: ignore

    eval_config: EvalConfig = yaml.safe_load(Path(eval_config).read_text())
    # load lists of privacy levels
    eps_list = eval_config["eps"] # type: ignore
    delta_sum_list = eval_config["delta"] # type: ignore

    # Prepare arguments for the worker function
    args = [
        (sbm_config, ds, eval_config, rng, float(eps), float(delta_sum))# type: ignore
        for sbm_config, ds, eps, delta_sum in
            product(
                fit_config["sbm"],
                data_config,
                eps_list,
                delta_sum_list,
            )
    ]

    if parallel:
        # Create a pool of workers
        n_workers = max(1, cpu_count() - 1)  # Leave one core free
        with Pool(n_workers) as pool:
            # Execute the worker function in parallel
            pool.starmap(_generate_and_evaluate, args)
    else:
        for sbm_config, ds, eval_config, rng, eps, delta_sum in args:
            _generate_and_evaluate(
                sbm_config=sbm_config,
                ds=ds,
                eval_config=eval_config,
                rng=rng,
                eps=eps,
                delta_sum=delta_sum,
            )

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fit_config", type=str, help="Path to the configuration file.")
    p.add_argument("--eval_config", type=str, help="Path to the configuration file.")
    p.add_argument("--parallel", action="store_true", help="Run in parallel leaving one core free.")
    args = p.parse_args()

    main(
        fit_config=args.fit_config,
        eval_config=args.eval_config,
        parallel=args.parallel,
    )