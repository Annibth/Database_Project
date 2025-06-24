import torch
from torch_geometric.nn import DiffPool
from torch_geometric.nn import GCNConv

class DiffPoolModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, embedding_channels, num_clusters, num_classes):
       super().__init__()
       
       self.gnn_embed_1 = GCNConv(in_channels, hidden_channels) 
       self.gnn_assign_1 = GCNConv(in_channels, num_clusters)
       
       self.diffpool = DiffPool(hidden_channels, num_clusters)
       