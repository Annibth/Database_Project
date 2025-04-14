import torch
import torch.nn.functional as F 
from torch.nn import Linear
from torch_geometric.nn import MessagePassing, global_mean_pool

class MPNNLayer (MessagePassing):
    
    def __init__(self, chan_in, chan_out):
        super(MPNNLayer, self).__init__(aggr='mean')
        
        self.linear = Linear(chan_in , chan_out)
        self.relu = torch.nn.ReLU()
        
    def forward(self, feat, edge):
        return self.propagate(edge, x=feat)
    
    def message(self, feat_neighbour):
        return self.relu(self.linear(feat_neighbour))
    
    def update(self, aggregated_msg, feat):
        return aggregated_msg + feat
    
    
class MPNN(torch.nn.Module):
    
    def __init__(self, chan_in, chan_hidden, chan_out):
        super(MPNN, self).__init__()
        
        self.layer1 = MPNNLayer(chan_in, chan_hidden)
        self.layer2 = MPNNLayer(chan_hidden,chan_hidden)
        self.layer3 = MPNNLayer(chan_hidden, chan_out)
        
        self.linear = Linear(chan_out, 1)
    
    def forward(self, feat, edge, batch):
        
        feat = self.layer1(feat, edge)
        feat = self.layer2(feat, edge)
        f_out = self.layer3(feat, edge)
        
        f_pool = global_mean_pool(f_out, batch)
        
        x = self.linear(f_pool)
        
        return x
