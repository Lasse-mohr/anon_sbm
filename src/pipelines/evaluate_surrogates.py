"""
    Generate surrogate SBM graphs and evaluate them against empirical data.
"""
import argparse
import yaml
import csv
from pathlib import Path
import numpy as np

from itertools import product
from multiprocessing import Pool, cpu_count

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


#######################
### Configuration Types 
#######################

def _generate_and_evaluate(sbm_config, ds, eval_config, rng):
    # Load empirical graph
    g = GraphLoader.load(
            Path(ds["path"]),
            force_undirected=sbm_config["force_undirected"], # type: ignore
        )
    emp = g.adjacency

    # load fitted model
    fit_folder_path = sbmfit_folderpath(
        base_dir=Path("results/sbm_fits"),
        sbm_config=sbm_config,# type: ignore
        data_spec=ds,
    )
    # load the sbm fit
    sbm_fit = SBMWriter.load(fit_folder_path)

    # check if metrics have been cached earlier
    out = surrogate_statistics_filename(
        base_dir=Path("results/surrogate_statistics"),
        eval_configs=eval_config,
        sbm_config=sbm_config,# type: ignore
        data_spec=ds,
    )

    out.parent.mkdir(exist_ok=True)
    if out.exists() and not eval_config["overwrite"]:
        return

    # Generate surrogates and campare metrics
    results = []
    for i in range(eval_config["n_surrogates"]):
        surr = sample_sbm_graph_from_fit(
            sbm_fit=sbm_fit,
            rng=rng,
        )
        surr = surr.adjacency

        row = {"dataset": ds["name"], "surrogate": f'surr_{i}'}
        for m in eval_config["metrics"]:
            row[m] = REGISTRY[m](emp, surr)

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

    # Prepare arguments for the worker function
    args = [
        (sbm_config, ds, eval_config, rng)
        for sbm_config, ds in product(fit_config["sbm"], data_config)
    ][::-1]
    if parallel:
        # Create a pool of workers
        n_workers = max(1, cpu_count() - 1)  # Leave one core free
        with Pool(n_workers) as pool:
            # Execute the worker function in parallel
            pool.starmap(_generate_and_evaluate, args)
    else:
        for sbm_config, ds in product(fit_config["sbm"], data_config):
            _generate_and_evaluate(sbm_config, ds, eval_config, rng)

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