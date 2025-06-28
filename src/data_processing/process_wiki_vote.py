"""
Script for converting wiki-voce data to simple edgelist format

change from format:
    # Directed graph (each unordered pair of nodes is saved once): Wiki-Vote.txt 
    # Wikipedia voting on promotion to administratorship (till January 2008). Directed edge A->B means user A voted on B becoming Wikipedia administrator.
    # Nodes: 7115 Edges: 103689
    # FromNodeId	ToNodeId
    30	1412
    30	3352
    30	5254

to format:
    30	1412
    30	3352
    30	5254

while removing self-loops and duplicate edges.
"""
from pathlib import Path

if __name__ == "__main__":
    file_path = Path("data/raw/wiki_vote/wiki-vote.txt")
    # read file
    with file_path.open("r") as f:
        lines = f.readlines()
    # process lines
    processed_lines = []
    edgeset = set()
    for index, line in enumerate(lines):
        if index >= 4:
            parts = line.split()
            edge = tuple(sorted(( int(parts[0]), int(parts[1]) )))

            if edge not in edgeset:
                edgeset.add(edge)
                # only keep the first two parts of the line
                # and ignore the third part (weight)
                # also ignore self-loops
                if parts[0] != parts[1]:
                    processed_lines.append(f"{parts[0]} {parts[1]}\n")

    # processed data goes in the "processed" folder in the parent-folder
    out_folder_path = Path("data/processed")
    out_file_path = out_folder_path / "wiki_vote.edgelist"

    # write to output file
    out_folder_path.mkdir(parents=True, exist_ok=True)
    with out_file_path.open("w") as f:
        f.writelines(processed_lines)
    print(f"Processed edgelist saved to {out_file_path}")