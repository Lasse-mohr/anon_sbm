from pathlib import Path
import subprocess, sys

def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

CFG_FIT = "configs/sbm_fit_k10.yml"
CFG_EVAL = "configs/surrogate_eval.yml"

if __name__ == "__main__":
    python = sys.executable

    run([python, "-m", "pipelines.fit_sbm",   CFG_FIT])
    run([python, "-m", "pipelines.generate_surrogates", CFG_EVAL])
    run([python, "-m", "pipelines.evaluate_surrogates", CFG_EVAL])