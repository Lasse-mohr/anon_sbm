# src/pipelines/fit_sbm.py
import yaml
import argparse
from pathlib import Path
from itertools import product

from line_profiler import profile

from time import time

import numpy as np
from tqdm import tqdm

from sbm.io import GraphLoader, SBMWriter
from sbm.block_assigner import AssignerConstructor
from sbm.model import SBMModel
from sbm.utils.logger import CSVLogger

from sbm.utils.pipeline_utils import (
    sbmfit_folderpath,
    FitConfig,
)

@profile
def main(fit_config: str): # type: ignore

    fit_config: FitConfig = yaml.safe_load(Path(fit_config).read_text())

    logging_config = fit_config["logging"][0] # type: ignore
    data_config = fit_config["datasets"] # type: ignore

    seed = fit_config['seed']
    rng = np.random.default_rng(seed)

    config_pairs = product(
        fit_config['sbm'], # type: ignore
        fit_config['datasets'], # type: ignore
    )

    iterator = tqdm(
        config_pairs,
        desc="Fitting SBM models",
        total=len(fit_config['sbm']) * len(fit_config['datasets'])
    )

    for sbm_config, ds in iterator:
        # prepare SBM model
        g = GraphLoader.load(
            Path(ds["path"]),
            force_undirected=bool(sbm_config["force_undirected"]), # type: ignore
            )
        assigner_const = AssignerConstructor(rng=rng)
        assigner = assigner_const.create_assigner(
            graph_data=g,
            min_block_size=int(sbm_config["min_block_size"]), # type: ignore
            init_method=sbm_config["init_method"], # type: ignore
            )

        block_data = assigner.compute_assignment()
        # check block sizes

        model = SBMModel(
            initial_blocks=block_data,
            rng=rng)
        # fit SBM model
        name = ds["name"] + "_".join(
            f"{k}_{v}" for k, v in sbm_config.items() # type: ignore
        )
        log_path = Path(logging_config['logging_folder']) / f"{name}.csv"

        tic = time()
        with CSVLogger(log_path, log_every=logging_config['log_every']) as logger:
            model.fit(
                min_block_size=sbm_config["min_block_size"], # type: ignore
                cooling_rate=sbm_config["cooling_rate"], # type: ignore
                logger=logger,
            )

        # save the fitted model
        fit = model.to_sbmfit()

        toc = time()
        print(f"Fitting {ds['name']} took {toc - tic:.2f} seconds")

        tic = time()
        fit_configs = sbm_config.copy() # type: ignore
        out_dir = sbmfit_folderpath(
            base_dir=Path("results/sbm_fits"),
            sbm_config=fit_configs,
            data_spec=ds, 
        )

        print(f'Out directory: {out_dir}')

        out_dir.mkdir(parents=True, exist_ok=True)
        SBMWriter.save(out_dir, fit)
        toc = time()
        print(f"Saving {ds['name']} took {toc - tic:.2f} seconds, nll = {fit.neg_loglike:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fit_config", type=str, help="Path to the configuration file.")
    args = p.parse_args()

    main(fit_config=args.fit_config)