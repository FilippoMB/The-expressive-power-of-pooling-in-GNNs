Official repository of the paper [The expressive power of pooling in Graph Neural Networks](https://arxiv.org/).

<img src="./figs/framework.png" width="700" height="150">

### The EXPWL1 dataset



### Execution
Run ````python main.py --pooling 'method'```` to perform graph classification on the EXPWL1 dataset, where ````'method'```` can be one of the following:

````python 
None, 'diffpool', 'mincut', 'dmon', 'edgepool', 'graclus', 'kmis', 'topk', 'panpool', 
'asapool', 'sagpool', 'dense-random', 'sparse-random', 'comp-graclus'
````

The code is based on [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/) and was tested on:
- torch v2.0.0
- torch_geometric v2.3.0

### Citation

Please, consider citing our paper if you use this code or the dataset in your research.
