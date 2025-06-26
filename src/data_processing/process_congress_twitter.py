"""
Script for converting congress data to simple edgelist format

change from format:
  0 4 {'weight': 0.002105263157894737}
  0 12 {'weight': 0.002105263157894737}
  0 18 {'weight': 0.002105263157894737}
  0 25 {'weight': 0.004210526315789474}

to format:
  0 4
  0 12
  0 18
  0 25

"""
from pathlib import Path

if __name__ == "__main__":
    file_path = Path("data/raw/congress_twitter/congress.edgelist")
    # read file
    with file_path.open("r") as f:
        lines = f.readlines()
    # process lines
    processed_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            processed_lines.append(f"{parts[0]} {parts[1]}\n")

    # processed data goes in the "processed" folder in the parent-folder
    out_folder_path = Path("data/processed")
    out_file_path = out_folder_path / "congress_twitter.edgelist"
    # write to output file
    out_folder_path.mkdir(parents=True, exist_ok=True)
    with out_file_path.open("w") as f:
        f.writelines(processed_lines)
    print(f"Processed edgelist saved to {out_file_path}")
