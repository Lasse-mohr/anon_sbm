from .shortest_path import shortest_path_distance
from .degree import degree_distance
from .clustering import clustering_distance

# registry maps a short name -> call-able
REGISTRY = {
    "shortest_path": shortest_path_distance,
    "degree": degree_distance,
    "clustering": clustering_distance,
}

