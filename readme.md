# PyCuda Tree-Child checker
This code checks whether a phylogenetic network is a tree-child network using pyCuda.
This is a linear-time computational problem on binary directed acyclic graphs, as it requires a "local" check for each node in the graph.
By doing this check for each node in a separate GPU thread, there is a substantial potential improvement in running time.

## Requirements
The code requires an installation of cuda on your system, plus the following python packages which can be installed with pip: phylox, pycuda, numpy.
Alternatively, you can install all requirements with conda/mamba. Note that the version of cuda has been pinned to work on a system with cuda version 12.9:
```
   mamba env update -f ./envs/pycuda-tc.source.yaml
   conda activate pycuda-tc
```

## Running the code
The code first creates a number of networks with parameters as set in the main function
 - n: the number of leaves of the network
 - max_k: the maximum number of reticulations (k) in the networks
 - step: the step to increase the number of reticulations (k) from 0 to max_k
 - iterations: the number of iterations per combination (n,k)
These networks are saved as Newick strings to separate files. Note that generating the networks takes a lot of time, because phylox has not been optimized to create large networks. The I/O also takes quite some time for larger networks!

After generating the networks, each network is read in again from file, and both CPU and GPU code is used to determine whether the network is tree-child.
Running times are logged in a csv file. Note that reading the network and converting to a GPU-usable format are not taken into account for the running times to make the comparison fair (assuming that normally, you'd already have your network loaded in the appropriate format whenever you work with it).

### Running commands
The code can simply be run with python:
```
  python ./pycuda_tree_child.py > ./output/test.out
```

On an LSF cluster, run the code with bsub and an appropriate amount of memory.
```
  bsub -o output/cluster.out -e output/cluster.err -M 128G -q bio-gpu-v100 "python ./pycuda_tree_child.py > ./output/test.out"
```

