[![arXiv](https://img.shields.io/badge/arXiv-2304.01575-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2304.01575)

Official repository of the paper *"The expressive power of pooling in Graph Neural Networks"* by F. M. Bianchi and V. Lachi.

In a nutshell, a graph pooling operator can be expressed as the composition of 3 functions:

- $\texttt{SEL}$: defines how to form the vertices of the coarsened graph;
- $\texttt{RED}$: computes the vertex features in the coarsened graph;
- $\texttt{CON}$: computes the edges in the coarsened graphs.

More details about the Select-Reduce-Connect framework can be found [here](https://arxiv.org/abs/2110.05292).

If certain conditions are met on the GNN layers before pooling, on the $\texttt{SEL}$, and the $\texttt{RED}$ functions, then enough information is preserved in the coarsened graph.
In particular, if two graphs $\mathcal{G}_ 1$ and $\mathcal{G}_ 2$ are WL-distinguishable, their coarsened versions $\mathcal{G}_ {1_P}$ and $\mathcal{G}_{2_P}$ will also be WL-dinstinguishable.

<img src="./figs/framework.png" width="523" height="235">

This repository contains the dataset and the code to test empirically the expressivity of existing graph pooling operators, which is their ability to generate coarsened graphs that are still distinguishable.

### Execution
Run ````python main.py --pooling 'method'```` to perform graph classification on the EXPWL1 dataset, where ````'method'```` can be one of the following:

````python 
None, 'diffpool', 'mincut', 'dmon', 'edgepool', 'graclus', 'kmis', 'topk', 'panpool', 
'asapool', 'sagpool', 'dense-random', 'sparse-random', 'comp-graclus'
````

The code is based on [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/) and was tested on:
- torch v2.0.0
- torch_geometric v2.3.0

### The EXPWL1 dataset
The dataset contains 1500 pairs of graphs $(\mathcal{G}_i, \mathcal{H_i})$, which are non-isomorphic and WL-1 distinguishable.
A GNN as powerful as the WL-1 test should achieve approximately 100\% accuracy on this dataset.
Two of the graph pairs in the dataset are displayed below.

---

<img src="./figs/ex1.png" width="397" height="197">

---

<img src="./figs/ex2.png" width="397" height="197">

---

The dataset can be downloaded [here](https://github.com/FilippoMB/The-expressive-power-of-pooling-in-GNNs/tree/main/data/EXPWL1) and the Pytorch Geometric loader is in [utils.py](https://github.com/FilippoMB/The-expressive-power-of-pooling-in-GNNs/blob/0a25de158c336acab697398951d6d3a0fec1c6cf/scripts/utils.py#L30).

### Citation

Please, consider citing our paper if you use this code or the dataset in your research:

````bibtex
@misc{bianchi2023expressive,
    title={The expressive power of pooling in Graph Neural Networks}, 
    author={Filippo Maria Bianchi and Veronica Lachi},
    year={2023},
    eprint={2304.01575},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
````
