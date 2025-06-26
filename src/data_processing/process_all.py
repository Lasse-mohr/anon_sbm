# src/data_processing/process_all.py
""" 
Script to run all processing steps of all datasets.
"""
import subprocess
import sys

def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

datasets = [
    "data_processing.process_congress_twitter",
    "data_processing.process_email_eu",
    "data_processing.process_caltech",
]

if __name__ == "__main__":
    python = sys.executable
    for dataset in datasets:
        run([python, "-m", dataset])