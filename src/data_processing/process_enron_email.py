"""
Script for converting enron-email data to simple edgelist format

change from format:
    # Directed graph (each unordered pair of nodes is saved once): Email-Enron.txt 
    # Enron email network (edge indicated that email was exchanged, undirected edges)
    # Nodes: 36692 Edges: 367662
    # FromNodeId	ToNodeId
    0	1
    1	0
    1	2
    1	3

to format:
    0	1
    1	0
    1	2
    1	3

while removing self-loops and duplicate edges.
"""
from pathlib import Path

if __name__ == "__main__":
    file_path = Path("data/raw/enron_email/email-Enron.txt")
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
                if parts[0] != parts[1]:
                    processed_lines.append(f"{parts[0]} {parts[1]}\n")

    # processed data goes in the "processed" folder in the parent-folder
    out_folder_path = Path("data/processed")
    out_file_path = out_folder_path / "enron_email.edgelist"

    # write to output file
    out_folder_path.mkdir(parents=True, exist_ok=True)
    with out_file_path.open("w") as f:
        f.writelines(processed_lines)
    print(f"Processed edgelist saved to {out_file_path}")