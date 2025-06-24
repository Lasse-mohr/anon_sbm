import argparse, yaml
from pathlib import Path
import numpy as np

from sbm.io import SBMWriter
from sbm.sampling import sample_sbm_graph_from_fit

def main(cfg):
    cfg = yaml.safe_load(Path(cfg).read_text())
    rng = np.random.default_rng(cfg["seed"])

    for ds in cfg["datasets"]:
        sbm_fit = SBMWriter.load(Path(ds["sbm_model"]))
        out_dir = Path("data/surrogates") / ds["name"]
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(cfg["n_surrogates"]):
            fn = out_dir / f"surr_{i:03d}.npz"
            if fn.exists() and not cfg["overwrite"]:
                continue

            sur_graph_data = sample_sbm_graph_from_fit(
                sbm_fit=sbm_fit,
                rng=rng,
            )

            adj = sur_graph_data.adjacency
            np.savez_compressed(fn,
                                data=adj.data,
                                indices=adj.indices,
                                indptr=adj.indptr,
                                shape=adj.shape, # type: ignore
                            )
            print("generated", fn)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cfg")
    main(p.parse_args().cfg)
