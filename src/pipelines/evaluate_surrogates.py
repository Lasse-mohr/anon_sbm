import argparse, yaml, csv
from pathlib import Path
import numpy as np
import scipy.sparse as sp

from metrics import REGISTRY
from sbm.io import GraphLoader

from sbm.utils.pipeline_utils import load_csr_npz

def main(cfg):
    cfg = yaml.safe_load(Path(cfg).read_text())

    for ds in cfg["datasets"]:
        results = []
        emp = GraphLoader.load(Path(ds["graph"])).adjacency
        surr_dir = Path("data/surrogates") / ds["name"]

        for surr_file in sorted(surr_dir.glob("surr_*.npz")):
            surr = load_csr_npz(surr_file)
            row = {"dataset": ds["name"], "surrogate": surr_file.stem}
            for m in cfg["metrics"]:
                row[m] = REGISTRY[m](emp, surr)
            results.append(row)
            print("evaluated", surr_file)

        out = Path("results") / f"surrogate_metrics.csv"
        out.parent.mkdir(exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader(); w.writerows(results)
        print("saved", out)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("cfg"); main(p.parse_args().cfg)
