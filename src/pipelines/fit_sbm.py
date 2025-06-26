# src/pipelines/fit_sbm.py
import yaml
import argparse
from pathlib import Path

import numpy as np

from sbm.io import GraphLoader, SBMWriter
from sbm.block_assigner import AssignerConstructor
from sbm.model import SBMModel
from sbm.utils.logger import CSVLogger

from sbm.utils.pipeline_utils import (
    sbmfit_folderpath,
    FitConfig,
    DatasetSpec,
    SBMConfig
)

def main(fit_config: str): # type: ignore

    fit_config: FitConfig = yaml.safe_load(Path(fit_config).read_text())

    logging_config = fit_config["logging"][0] # type: ignore
    data_config = fit_config["datasets"] # type: ignore

    seed = fit_config['seed']
    rng = np.random.default_rng(seed)

    for sbm_config in fit_config['sbm']:
        for ds in data_config:
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
            model = SBMModel(block_data, rng=rng)

            # fit SBM model
            name = ds["name"] + "_".join(
                f"{k}_{v}" for k, v in sbm_config.items() # type: ignore
            )
            log_path = Path(logging_config['logging_folder']) / f"{name}.csv"

            with CSVLogger(log_path, log_every=logging_config['log_every']) as logger:
                model.fit(num_iterations=sbm_config["n_iter"], # type: ignore
                          min_block_size=sbm_config["min_block_size"], # type: ignore
                          initial_temperature=sbm_config["temperature"], # type: ignore
                          cooling_rate=sbm_config["cooling_rate"], # type: ignore
                          logger=logger,
                          )

            # save the fitted model
            fit = model.to_sbmfit()

            fit_configs = sbm_config.copy() # type: ignore
            out_dir = sbmfit_folderpath(
                base_dir=Path("data/sbm_fits"),
                sbm_config=fit_configs,
                data_spec=ds, 
            )

            out_dir.mkdir(parents=True, exist_ok=True)
            SBMWriter.save(out_dir, fit)
            print(f"{ds['name']} completed, nll = {fit.neg_loglike:.2f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fit_config", type=str, help="Path to the configuration file.")
    args = p.parse_args()

    main(fit_config=args.fit_config)