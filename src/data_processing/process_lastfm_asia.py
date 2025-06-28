"""
Script for converting lastfm-asia data to simple edgelist format

remove the header and remove self-loops and duplicate edges.
"""
from pathlib import Path

if __name__ == "__main__":
    file_path = Path("data/raw/lastfm_asia/lastfm_asia_edges.csv")
    # read file
    with file_path.open("r") as f:
        lines = f.readlines()
    # process lines
    processed_lines = []
    edgeset = set()
    for index, line in enumerate(lines):
        if index >= 1:
            parts = line.split(",")
            edge = tuple(sorted(( int(parts[0]), int(parts[1]) )))

            if edge not in edgeset:
                edgeset.add(edge)
                if len(parts) >= 1 and parts[0] != parts[1]:
                    processed_lines.append(f"{parts[0]} {parts[1]}\n")

    # processed data goes in the "processed" folder in the parent-folder
    out_folder_path = Path("data/processed")
    out_file_path = out_folder_path / "lastfm_asia.edgelist"

    # write to output file
    out_folder_path.mkdir(parents=True, exist_ok=True)
    with out_file_path.open("w") as f:
        f.writelines(processed_lines)
    print(f"Processed edgelist saved to {out_file_path}")