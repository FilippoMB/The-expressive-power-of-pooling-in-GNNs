from math import ceil
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GINConv, MLP, DenseGINConv, PANConv
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import TopKPooling, PANPooling, SAGPooling, ASAPooling, EdgePooling, graclus
from torch_geometric.nn import dense_mincut_pool, dense_diff_pool, DMoNPooling
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import to_dense_adj, to_dense_batch

from scripts.sum_pool import sum_pool
from scripts.pooling.kmis.kmis_pool import KMISPooling
from scripts.pooling.rnd_sparse import RndSparse
from scripts.utils import batched_negative_edges


class GIN_Pool_Net(torch.nn.Module):
    def __init__(self, 
                 in_channels,           # Size of node features
                 out_channels,          # Number of classes
                 num_layers_pre=1,      # Number of GIN layers before pooling
                 num_layers_post=1,     # Number of GIN layers after pooling
                 hidden_channels=64,    # Dimensionality of node embeddings
                 norm=True,             # Normalise Layers in the GIN MLP
                 activation='ELU',      # Activation of the MLP in GIN 
                 average_nodes=None,    # Needed for dense pooling methods
                 max_nodes=None,        # Needed for random pool
                 pooling=None,          # Pooling method
                 pool_ratio=0.1,        # Ratio = nodes_after_pool/nodes_before_pool
                 ):
        super(GIN_Pool_Net, self).__init__()
        
        self.num_layers_pre = num_layers_pre
        self.num_layers_post = num_layers_post
        self.hidden_channels = hidden_channels
        self.act = activation_resolver(activation)
        self.pooling = pooling
        self.pool_ratio = pool_ratio
  
        # Pre-pooling block            
        self.conv_layers_pre = torch.nn.ModuleList()
        if pooling=='panpool':
            for _ in range(num_layers_pre):
                self.conv_layers_pre.append(PANConv(in_channels, hidden_channels, filter_size=2))
                in_channels = hidden_channels
        else:
            for _ in range(num_layers_pre):
                mlp = MLP([in_channels, hidden_channels, hidden_channels], act=activation)
                self.conv_layers_pre.append(GINConv(nn=mlp, train_eps=False))
                in_channels = hidden_channels
                        
        # Pooling block
        pooled_nodes = ceil(pool_ratio * average_nodes)
        if pooling in ['diffpool','mincut']:
            self.pool = Linear(hidden_channels, pooled_nodes)
        elif pooling=='dmon':
            self.pool = DMoNPooling(hidden_channels, pooled_nodes)
        elif pooling=='dense-random':
            self.s_rnd = torch.randn(max_nodes, pooled_nodes)
            self.s_rnd.requires_grad = False
        elif pooling=='topk':
            self.pool = TopKPooling(hidden_channels, ratio=pool_ratio)
        elif pooling=='panpool':
            self.pool = PANPooling(hidden_channels, ratio=pool_ratio)
        elif pooling=='sagpool':
            self.pool = SAGPooling(hidden_channels, ratio=pool_ratio)
        elif pooling=='asapool':
            self.pool = ASAPooling(hidden_channels, ratio=pool_ratio)  
        elif pooling=='edgepool':
            self.pool = EdgePooling(hidden_channels)
        elif pooling=='kmis':
            self.pool = KMISPooling(hidden_channels, k=5, aggr_x='sum')
        elif pooling in ['graclus', 'comp-graclus']:
            pass
        elif pooling=='sparse-random':
            self.pool = RndSparse(pool_ratio, max_nodes)
        else:
            assert pooling==None
        
        # Post-pooling block
        self.conv_layers_post = torch.nn.ModuleList()
        for _ in range(num_layers_post):
            mlp = MLP([hidden_channels, hidden_channels, hidden_channels], act=activation, norm=None)
            if pooling in ['diffpool','mincut','dmon','dense-random']:
                self.conv_layers_post.append(DenseGINConv(nn=mlp, train_eps=False))                
            else:
                self.conv_layers_post.append(GINConv(nn=mlp, train_eps=False))

        # Readout
        self.mlp = MLP([hidden_channels, hidden_channels, hidden_channels//2, out_channels], 
                        act=activation,
                        norm=None,
                        dropout=0.5)


    def reset_parameters(self):
        for (name, module) in self._modules.items():
            try:
                module.reset_parameters()
            except AttributeError:
                if name != 'act':
                    for x in module:
                        x.reset_parameters()


    def forward(self, data):
        x = data.x    
        adj = data.edge_index
        batch = data.batch

        ### pre-pooling block
        if self.pooling=='panpool':
            M = adj
            for layer in self.conv_layers_pre:  
                x, M = layer(x, M)
                x = self.act(x)
        else:
            for layer in self.conv_layers_pre:  
                x = self.act(layer(x, adj))
    
        ### pooling block
        if self.pooling in ['diffpool','mincut','dmon','dense-random']:
            x, mask = to_dense_batch(x, batch)
            adj = to_dense_adj(adj, batch)
            if self.pooling=='diffpool':
                s = self.pool(x)
                x, adj, l1, l2 = dense_diff_pool(x, adj, s, mask)
                aux_loss = 0.1*l1 + 0.1*l2
            elif self.pooling=='mincut':
                s = self.pool(x)
                x, adj, l1, l2 = dense_mincut_pool(x, adj, s, mask)
                aux_loss = 0.5*l1 + l2
            elif self.pooling=='dmon':  
                _, x, adj, l1, l2, l3 = self.pool(x, adj, mask)
                aux_loss = 0.3*l1 + 0.3*l2 + 0.3*l3
            elif self.pooling=='dense-random':
                s = self.s_rnd[:x.size(1), :].unsqueeze(dim=0).expand(x.size(0), -1, -1).to(x.device)
                x, adj, _, _ = dense_diff_pool(x, adj, s, mask)
        elif self.pooling in ['topk', 'sagpool', 'sparse-random']:
            x, adj, _, batch, _, _ = self.pool(x, adj, edge_attr=None, batch=batch)
        elif self.pooling=='asapool':
            x, adj, _, batch, _ = self.pool(x, adj, batch=batch)
        elif self.pooling=='panpool':
            x, adj, _, batch, _, _ = self.pool(x, M, batch=batch)
        elif self.pooling=='edgepool':
            x, adj, batch, _ = self.pool(x, adj, batch=batch)
        elif self.pooling=='kmis':
            x, adj, _, batch, _, _ = self.pool(x, adj, None, batch=batch)
        elif self.pooling in ['graclus', 'comp-graclus']:
            data.x = x    
            if self.pooling == 'graclus':
                cluster = graclus(adj, num_nodes=data.x.size(0))
            else:
                complement = batched_negative_edges(edge_index=adj, batch=batch, force_undirected=True)
                cluster = graclus(complement, num_nodes=data.x.size(0))
            data = sum_pool(cluster, data)
            x = data.x    
            adj = data.edge_index
            batch = data.batch
        elif self.pooling==None:
            pass
        else:
            raise KeyError("unrecognized pooling method")
                
        ### post-pooling block
        for layer in self.conv_layers_post:  
            x = self.act(layer(x, adj))

        ### readout
        if self.pooling in ['diffpool','mincut','dmon','dense-random']:
            x = torch.sum(x, dim=1)
        else:
            x = global_add_pool(x, batch)
        x = self.mlp(x)
        
        if 'aux_loss' not in locals():
            aux_loss=0
        return F.log_softmax(x, dim=-1), aux_loss