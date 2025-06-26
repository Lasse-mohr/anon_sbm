""" 
Script to run fitting, generation, and evaluation pipelines in sequence.
"""
import subprocess
import sys

def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

CFG_FIT = "configs/sbm_fit_block_size_experiments.yml"
CFG_EVAL = "configs/surrogate_eval.yml"

if __name__ == "__main__":
    python = sys.executable

    run([
        python, "-m",
        "pipelines.fit_sbm",
        "--fit_config", CFG_FIT
    ])
    run([
        python, "-m",
        "pipelines.generate_and_evaluate_surrogates",
        "--fit_config", CFG_FIT,
        "--eval_config", CFG_EVAL,
    ])