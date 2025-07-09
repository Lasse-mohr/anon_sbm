from .shortest_path import shortest_path_distance, avg_path_length_difference
from .degree import degree_distance
from .clustering import clustering_distance, avg_clustering_difference
from .spectral import eigen_val_distance, centrality_distance
from .embedding import embedding_node2vec_ip_emd, embedding_prone_ip_emd
from .community import infomap_codelength_difference, leiden_modularity_difference
from .assortativity import assortativity_difference



# registry maps a short name -> call-able
REGISTRY = {
    "shortest_path": shortest_path_distance,
    "avg_path_length": avg_path_length_difference,
    "degree": degree_distance,
    "clustering": clustering_distance,
    "avg_clustering": avg_clustering_difference,
    "eigen_val": eigen_val_distance,
    "eigen_centrality": centrality_distance,
    "infomap": infomap_codelength_difference,
    "leiden": leiden_modularity_difference,
    "assortativity": assortativity_difference,
    "embedding_node2vec": embedding_node2vec_ip_emd,
    "embedding_prone": embedding_prone_ip_emd,
}

