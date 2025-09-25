% PyCuda Tree-Child
% Remie Janssen
% September 22, 2025

# Background

## Phylogenetic network

::: {.columns}
:::: {.column width="50%}

- Directed acyclic graph
- Single root (in-0, out-1)
- Labeled leaves (in-1, out-0)
- Tree nodes (in-1, out-2)
- Reticulations (in-2, out-1)

::::
:::: {.column width="50%}

![phylogenetic network](images/network.svg)

::::
:::

## Tree child network
::: {.columns}
:::: {.column width="50%}

 - Class of networks
 - Computationally favorable structure
 - "each non-leaf node has at least one tree-node as successor"
 - Equivalently: No stacks of reticulations and no W-shapes

::::
:::: {.column width="50%}

![Stack and W-shape](images/stack_and_w.svg)

::::
:::


# Implementation

## PyCuda

- Cuda wrapper for Python
- Still have to write device/global cuda code
- Implicit Memcpy using numpy data formats
- Cuda code in .cu file, loaded in python

## GPU Graph encoding
::: {.columns}
:::: {.column width="50%}

- Graph: in array
- Nodes: non-negative ints
- in- and out-neighbours per node
- "-1" for non-neighbour (here "-")

::::
:::: {.column width="50%}

![Graph encoding](images/graph_encoding.svg)

::::
:::
## GPU code

- Each thread checks one node
- Check whether it is a leaf
- Check if it has any tree-node children
- If not as leaf, and no tree-node children, set output to "not tree-child"


# Experiments

## Setup
::: {.columns}
:::: {.column width="50%}

- Create random tree with *n* leaves
- Add *k* reticulation edges
- *10* iterations per *(n,k)*

::::
:::: {.column width="50%}

![Network generation](images/generate.svg)

::::
:::

## Results
::: {.columns}
:::: {.column width="50%}

- GPU: Fairly constant running time
- CPU: Running time increases with n, and decreases with k
- CPU stops when it finds a stack or W-shape

::::
:::: {.column width="50%}
::::: {.r-stack}
:::::: {.fragment .current-visible fragment-index=0}

![Results n=100 -> 3000, k=1000](images/results_inc_n.png)

::::::
:::::: {.fragment .current-visible fragment-index=1}

![Results n=3000, k=0 -> 1000](images/results.png)

::::::
:::::
::::
:::

## Discussion

- As expected: GPU only useful for huge networks with few reticulations
- May be useful for other phylogenetic network problems
