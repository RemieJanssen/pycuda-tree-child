import pycuda.autoinit
import pycuda.driver as drv
import numpy
import networkx as nx
import datetime
import os

from phylox.generators.lgt import generate_network_lgt
from phylox.classes.dinetwork import is_tree_child
from phylox.constants import LABEL_ATTR
from phylox.dinetwork import DiNetwork

from pycuda.compiler import SourceModule

with open("tree_child.cu") as f:
    tree_child_source_code = f.read()

mod = SourceModule(tree_child_source_code)
device_is_tree_child = mod.get_function("is_tree_child")

def network_to_adjacency_list(network, max_degree=2):
    relabeled_network  = nx.convert_node_labels_to_integers(network)
    node_list = numpy.full(len(relabeled_network.nodes)*2*max_degree, -1, numpy.int32)
    for node in relabeled_network.nodes():
        in_neighbour_index = 2*node*max_degree
        for i, neighbour in enumerate(relabeled_network.predecessors(node)):
            node_list[in_neighbour_index+i] = neighbour
        out_neighbour_index = (2*node+1)*max_degree
        for i, neighbour in enumerate(relabeled_network.successors(node)):
            node_list[out_neighbour_index+i] = neighbour
    return node_list

def label_leaves(network):
    for i, l in enumerate(network.leaves):
        network.nodes[l][LABEL_ATTR] = i


def gpu_is_tree_child(nodes_adjacency_list, max_number_of_neighbours, number_of_nodes):
    result = numpy.zeros(1, dtype=numpy.int32)
    max_number_of_neighbours = numpy.int32(max_number_of_neighbours)

    MAX_NUMBER_OF_THREADS_PER_BLOCK = 512
    blocks = int(number_of_nodes / MAX_NUMBER_OF_THREADS_PER_BLOCK) + 1
    threads_per_block = min(MAX_NUMBER_OF_THREADS_PER_BLOCK, number_of_nodes)
    device_is_tree_child(
        max_number_of_neighbours, numpy.int32(number_of_nodes), drv.In(nodes_adjacency_list), drv.Out(result),
        block=(threads_per_block,1,1), grid=(blocks,1))
    return result[0] != 1


def generate_networks(networks_folder, n, max_k, step, iterations):
    for k in range(0,max_k+1,step):
        for i in range(iterations):
            print(f"generating network with {n} leaves and {k} reticulations")
            start_generating = datetime.datetime.now()
            network = generate_network_lgt(n,k,0.5,0.5)
            label_leaves(network)
            print(network)
            end_generating = datetime.datetime.now()
            print(f"generating took {end_generating - start_generating}")
            with open(os.path.join(networks_folder, f"{n}_{k}_{i}"), "w") as f:
                f.write(network.newick())

if __name__ == "__main__":
    n = 3
    max_k = 3
    step = 1
    iterations = 2
    output_folder = "./output/"
    networks_folder = os.path.join(output_folder, "networks")
    output_file = os.path.join(output_folder, "results.csv")
    os.makedirs(networks_folder, exist_ok=True)
    # generate_networks(networks_folder, n, max_k, step, iterations)

    with open(output_file, "w+") as f:
        f.write(f"n,k,i,gpu_tc,cpu_tc,gpu_time,cpu_time\r\n")

    for network_file in os.listdir(networks_folder):
        with open(os.path.join(networks_folder, network_file), "r") as f:
            newick = f.read()
        n,k,i = network_file.split("_")
        network = DiNetwork.from_newick(newick)
        nodes_adjacency_list = network_to_adjacency_list(network, max_degree=2)
        number_of_nodes = len(network.nodes)

        start_cpu_calculation = datetime.datetime.now()
        cpu_is_tc = is_tree_child(network)
        end_cpu_calculation = datetime.datetime.now()
        cpu_time = (end_cpu_calculation- start_cpu_calculation).total_seconds()

        start_gpu_calculation = datetime.datetime.now()
        gpu_is_tc = gpu_is_tree_child(nodes_adjacency_list, 2, number_of_nodes)
        end_gpu_calculation = datetime.datetime.now()
        gpu_time = (end_gpu_calculation-start_gpu_calculation).total_seconds()

        del network
        with open(output_file, "a+") as f:
            f.write(f"{n},{k},{i},{bool(gpu_is_tc)},{cpu_is_tc},{gpu_time},{cpu_time}\r\n")
