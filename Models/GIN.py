import torch
import torch.nn.functional as F                     #
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GINConv, global_mean_pool

class GIN(torch.nn.Module):
    
    def __init__(self, dim_in, dim_hidden, dim_out):
        super.__init__()
        
        # First Layer
        self.conv1 = GINConv(Sequential(Linear(dim_in,dim_hidden),ReLU(),Linear(dim_hidden,dim_hidden)))
        
        # further Layers ...
        
        # Output Layer, auf Outpu-Dimension reduziert
        self.out = Linear(dim_hidden,dim_out)
        
    def forward(self, feat, edge, batch):
        
        feat = self.conv1(feat, edge) #feat durch GINConv-Layer, Verarebitugn plus Nachbarinformation
        
        # further Layers
        
        feat = F.relu(feat) #nichjtlineare Aktivierung
        
        # Aggregieren zu Graph-Embedding (weitere Funktionen)
        graph = global_mean_pool(feat, batch)
        
        #Output
        x = self.out(graph)
        return x
        
        