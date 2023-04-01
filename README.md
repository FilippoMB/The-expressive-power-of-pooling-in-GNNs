Official repository of the paper [The expressive power of pooling in Graph Neural Networks](https://arxiv.org/).

The code is based on [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/).

### Execution
Run ````python main.py --pooling 'method'```` to perform graph classification on the EXPWL1 dataset, where ````'method'```` can be one of the following:

````python 
None, 'diffpool', 'mincut', 'dmon', 'edgepool', 'graclus', 'kmis', 'topk', 'panpool', 
'asapool', 'sagpool', 'dense-random', 'sparse-random', 'comp-graclus'
````

The code was tested on:
- torch v2.0.0
- torch_geometric v2.3.0

### Citation

Please, consider citing our paper if you use this code or the dataset in your research.
