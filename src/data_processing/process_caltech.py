"""
Script for converting caltech-facebook data to simple edgelist format

change from format:
    %MatrixMarket matrix coordinate pattern symmetric 
    769 769 16656
    5 1
    31 1

to format:
    769 769 16656
    5 1
    31 1

"""
from pathlib import Path

if __name__ == "__main__":
    file_path = Path("data/raw/socfb-Caltech36/socfb-Caltech36.mtx")
    # read file
    with file_path.open("r") as f:
        lines = f.readlines()
    # process lines
    processed_lines = []

    # skip first two lines (Matrix Market header)
    for idx, line in enumerate(lines):
        if idx < 2:
            continue

        parts = line.split()
        if len(parts) >= 2:
            processed_lines.append(f"{parts[0]} {parts[1]}\n")

    # processed data goes in the "processed" folder in the parent-folder
    out_folder_path = Path("data/processed")
    out_file_path = out_folder_path / "caltech_fb.edgelist"
    # write to output file
    out_folder_path.mkdir(parents=True, exist_ok=True)
    with out_file_path.open("w") as f:
        f.writelines(processed_lines)

    print(f"Processed edgelist saved to {out_file_path}")
