"""
Script for converting email-eu data to simple edgelist format

change from format:
    582 364 0
    168 472 2797
    168 912 3304
    2 790 4523

to format:
    582 364
    168 472
    168 912
    2 790

while removing self-loops and duplicate edges.
"""
from pathlib import Path

if __name__ == "__main__":
    file_path = Path("data/raw/email_eu/email-Eu-core-temporal.txt")
    # read file
    with file_path.open("r") as f:
        lines = f.readlines()
    # process lines
    processed_lines = []
    edgeset = set()
    for line in lines:
        parts = line.split()
        edge = tuple(sorted(( int(parts[0]), int(parts[1]) )))

        if edge not in edgeset:
            edgeset.add(edge)
            # only keep the first two parts of the line
            # and ignore the third part (weight)
            # also ignore self-loops
            if len(parts) >= 2 and parts[0] != parts[1]:
                processed_lines.append(f"{parts[0]} {parts[1]}\n")

    # processed data goes in the "processed" folder in the parent-folder
    out_folder_path = Path("data/processed")
    out_file_path = out_folder_path / "eu_email.edgelist"

    # write to output file
    out_folder_path.mkdir(parents=True, exist_ok=True)
    with out_file_path.open("w") as f:
        f.writelines(processed_lines)
    print(f"Processed edgelist saved to {out_file_path}")