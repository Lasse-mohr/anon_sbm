# src/pipelines/fit_sbm.py
import yaml
import argparse
import time
from pathlib import Path

import numpy as np

from sbm.io import GraphLoader, SBMWriter
from sbm.block_assigner import MetisBlockAssigner
from sbm.model import SBMModel
from sbm.utils.logger import CSVLogger

def main(cfg_path: str):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    for ds in cfg["datasets"]:
        seed = cfg['seed']
        rng = np.random.default_rng(seed)

        # prepare SBM model
        g = GraphLoader.load(
            Path(ds["path"]),
            force_undirected=cfg["force_undirected"],
            )
        assigner = MetisBlockAssigner(
            graph_data=g,
            min_block_size=cfg["min_block_size"],
            rng=rng,
            )

        block_data = assigner.compute_assignment()
        model = SBMModel(block_data, rng=rng)

        # fit SBM model

        name = (
            f"{ds["name"]}_"
            f"{cfg['n_iter']}_iter_"
            f"{cfg['min_block_size']}_minbs_"
        )
        log_path = Path(cfg['logging_folder']) / f"{name}.csv"

        tic = time.time()
        with CSVLogger(log_path, log_every=cfg['log_every']) as logger:
            model.fit(num_iterations=cfg["n_iter"],
                      min_block_size=cfg["min_block_size"],
                      initial_temperature=cfg["temperature"],
                      cooling_rate=cfg["cooling_rate"],
                      logger=logger,
                      )
        toc = time.time()

        # save the fitted model
        fit = model.to_sbmfit({"dataset": ds["name"], "fit_seconds": toc-tic})
        out_dir = Path("models/sbm")/ds["name"]/f'min_group_size_{cfg["min_block_size"]}'
        out_dir.mkdir(parents=True, exist_ok=True)
        SBMWriter.save(out_dir, fit)
        print(f"{ds['name']} completed, nll = {fit.neg_loglike:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", help="YAML config")
    main(parser.parse_args().cfg)
