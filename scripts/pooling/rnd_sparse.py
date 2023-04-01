from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from torch_geometric.utils import softmax
from torch_geometric.nn.pool.topk_pool import topk, filter_adj


class RndSparse(torch.nn.Module):
    r"""A sparse pooling layer that randomly selects the vertices to keep
    
    Args:
        in_channels (int): Size of each input sample.
        ratio (float or int): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`, or the value
            of :math:`k` itself, depending on whether the type of :obj:`ratio`
            is :obj:`float` or :obj:`int`.
        max_nodes (int): The maximum number of nodes that can be found in a batch
    """
    def __init__(
        self,
        ratio: Union[int, float] = 0.5,
        max_nodes=None
    ):
        super().__init__()

        self.ratio = ratio
        self.min_score = None
        self.score = torch.randn(max_nodes)


    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor, Tensor, Tensor]:

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
        score = softmax(self.score[:x.size(0)].to(x.device), batch)

        perm = topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score[perm]
