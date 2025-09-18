__device__ bool is_retic(int max_number_of_neighbors, int *nodes, int node_index)
{
    int number_of_in_neighbours = 0;
    for (int in_nb_index = (2 * node_index) * max_number_of_neighbors; in_nb_index < (2 * node_index + 1) * max_number_of_neighbors; in_nb_index++){
        if (nodes[in_nb_index] > -1){
            number_of_in_neighbours++;
        }
    }
    return (number_of_in_neighbours > 1);
}

__global__ void is_tree_child(int max_number_of_neighbors, int number_of_nodes, int *nodes, int *result)
{
    const int node_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (node_index >= number_of_nodes){
        return;
    }
    bool node_is_tree_child = false;
    bool has_out_neighbor = false;
    for (int out_nb_index = (2 * node_index + 1) * max_number_of_neighbors; out_nb_index < (2 * node_index + 2) * max_number_of_neighbors; out_nb_index++){
        if (nodes[out_nb_index] == -1){
            continue;
        }
        has_out_neighbor = true;
        if (! is_retic(max_number_of_neighbors, nodes, nodes[out_nb_index])){
            node_is_tree_child = true;
        }
    }
    if (has_out_neighbor && !node_is_tree_child){
        result[0] = 1;
    }
}