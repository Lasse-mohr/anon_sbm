from typing import Dict, Union, TypedDict, List, Literal
from pathlib import Path
import numpy as np
import scipy.sparse as sp



def load_csr_npz(fn: Path):
    with np.load(fn) as z:
        return sp.csr_matrix(
            (z["data"], z["indices"], z["indptr"]),
            shape=z["shape"]
        )

InitMethodName = Literal["metis", "random", "ProneKMeans"]

class DatasetSpec(TypedDict):
    name: str
    path: str

class SBMConfig(TypedDict):
    force_undirected: bool
    min_block_size: int
    n_iter: int
    temperature: float
    cooling_rate: float
    init_method: InitMethodName

class LoggingConfig(TypedDict):
    logging_folder: str
    log_every: int

class FitConfig(TypedDict):
    seed: int
    sbm: SBMConfig
    logging: LoggingConfig
    datasets: List[DatasetSpec]

class EvalConfig(TypedDict):
    n_surrogates: int
    overwrite: bool
    metrics: List[str]

def clean_filename(name: str) -> str:
    """
    Clean the name of all special characters and spaces, replacing them with underscores.
    """

    name = name.replace(":", "_")
    name = name.replace(".", "_")
    name = name.replace(",", "_")

    return name

def sbmfit_folderpath(
    base_dir: Path,
    sbm_config: SBMConfig,
    data_spec: DatasetSpec,
) -> Path:
    """
    Generate the folderpath for storing a fitted SBM model based on the fit configuration.

    Filename is created by unrolling the fit_config dictionary, using all fields and their values. 

    :param name: Name of the dataset. 
    """

    folder_name = data_spec["name"] + "_" + "_".join(
        f"{k}_{v}" for k, v in sorted(sbm_config.items())
    )
    folder_name = clean_filename(folder_name)
    return base_dir / f"sbm_fit_{folder_name}"

def surrogate_statistics_filename(
    base_dir: Path,
    eval_configs: EvalConfig,
    sbm_config: SBMConfig,
    data_spec: DatasetSpec,
) -> Path:
    """
    Generate the folfor surrogate statistics based on evaluation and fit configurations.

    :param eval_configs: Evaluation configuration dictionary.
    :param fit_config: Fit configuration dictionary.
    :return: Path object representing the filename.
    """

    file_name = (
        f"{data_spec['name']}_"
        f"surrogates_{eval_configs['n_surrogates']}_"
            #f"{'_'.join(eval_configs['metrics'])}_"
        #f"{'_'.join(f'{k}_{v}' for k, v in sorted(sbm_config.items()))}"
    )
    file_name = clean_filename(file_name)

    return base_dir / f"{file_name}.csv"

def dataset_filepath(
    base_dir: Path,
    dataset_name: str,
)-> Path:
    """
    Generate the filepath for a dataset based on its name.

    :param base_dir: Base directory where datasets are stored.
    :param dataset_name: Name of the dataset.
    :return: Path object representing the dataset file path.
    """
    dataset_name = clean_filename(dataset_name)

    return base_dir / f"{dataset_name}.npz"



##### Helper functions #####
def fit_config_to_dicts(fit_config: FitConfig) -> List[dict[str, str]]:
    """ 
    Convert FitConfig to a list of DatasetSpec dictionaries.
    One dictionary per dataset.
    """
    configs = [
        {
            "name": ds["name"],
            "path": ds["path"],
            **fit_config["sbm"],
        }
        for ds in fit_config["datasets"]
    ]

    for config in configs:
        if "seed" in config:
            config["seed"] = str(fit_config["seed"])

    return configs

def eval_config_to_dict(eval_config: EvalConfig) -> dict[str, str]:
    """
    Convert EvalConfig to a dictionary.
    """
    return {
        "n_surrogates": str(eval_config["n_surrogates"]),
        "metrics": "_".join(
                eval_config["metrics"]    
            )
    }