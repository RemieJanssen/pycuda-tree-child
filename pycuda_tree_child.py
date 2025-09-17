import pycuda.autoinit
import pycuda.driver as drv
import numpy
import networkx as nx
import phylox as px
import datetime

from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void is_tree_child(int max_number_of_neighbors, int *nodes, bool result)
{
    const int node_index = threadIdx.x;
    bool node_is_tree_child = false;
    bool has_out_neighbor = false;
    for (int out_nb_index = (2 * node_index + 1) * max_number_of_neighbors; out_nb_index < (2 * node_index + 2) * max_number_of_neighbors); out_nb_index++){
        if (out_nb_index == -1){
            continue;
        }
        has_out_neighbor = true;
        if (!is_retic(max_number_of_neighbors, nodes, out_nb_index){
            node_is_tree_child = true;
        }
    }
    if (node_is_tree_child or !has_out_neighbor){
        result = true;
    }
}

__device__ bool is_retic(int max_number_of_neighbors, int *nodes, int node_index)
{
    int number_of_in_neighbours = 0;
    for (int in_nb_index = (2 * node_index) * max_number_of_neighbors; in_nb_index < (2 * node_index + 1) * max_number_of_neighbors); in_nb_index++){
        if (in_nb_index != -1){
            number_of_in_neighbours++;
    return number_of_in_neighbours > 1;
}
""")


device_is_tree_child = mod.get_function("is_tree_child")

def network_to_adjacency_list(network, max_degree=2):
    relabeled_network  = nx.convert_node_labels_to_integers(network)
    node_list = numpy.full(len(relabeled_network.nodes)*2*max_degree, -1)
    for node in relabeled_network.nodes():
        in_neighbour_index = 2*node*max_degree
        for i, neighbour in relabeled_network.predecessors(node):
            node_list[in_neighbour_index+i] = neighbour
        out_neighbour_index = (2*node+2)*max_degree
        for i, neighbour in relabeled_network.successors(node):
            node_list[out_neighbour_index+i] = neighbour
    return node_list


def is_tree_child(network):
    result = False
    max_numer_of_neighbours = 2
    nodes = network_to_adjacency_list(network, max_degree=max_numer_of_neighbours)
    # a = numpy.random.randn(400).astype(numpy.int32)
    # b = numpy.random.randn(400).astype(numpy.int32)

    # dest = numpy.zeros_like([0])

    device_is_tree_child(
        drv.In(max_numer_of_neighbours), drv.In(nodes), drv.Out(result),
        block=(len(network.nodes()),1,1), grid=(1,1))
    return result


if __name__ == "__main__":
    network = px.generators.lgt.base.generate_network_lgt_conditional(100,10,0.1)
    print(network)
    print(datetime.datetime.now())
    print(is_tree_child(network))
    print(datetime.datetime.now())
