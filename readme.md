```
   mamba env update -f ./envs/pycuda-tc.source.yaml
```

```
  bsub -o output/cluster.out -e output/cluster.err -q bio-gpu-v100 "python ./pycuda_tree_child.py > ./output/test.out"
```