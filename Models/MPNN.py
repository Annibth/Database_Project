import torch
import torch.nn.functional as F 
from torch.nn import Linear
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data
from torch.optim import Adam

class MPNNLayer (MessagePassing):
    
    def __init__(self, chan_in, chan_out):
        super(MPNNLayer, self).__init__(aggr='mean')
        
        self.linear = Linear(chan_in , chan_out)
        self.relu = torch.nn.ReLU()
        self.residual = Linear(chan_in, chan_out)
        
    def forward(self, x, edge_index):
        return self.propagate(edge_index, size = (x.size(0), x.size(0)), x=x)
    
    def message(self, x_j):
        return self.relu(self.linear(x_j))
         
    
    def update(self, aggr_out, x):
        return aggr_out + self.residual(x)
    
    
class MPNN(torch.nn.Module):
    
    def __init__(self, chan_in, chan_hidden, chan_out):
        super(MPNN, self).__init__()
        
        self.layer1 = MPNNLayer(chan_in, chan_hidden)
        self.layer2 = MPNNLayer(chan_hidden,chan_hidden)
        self.layer3 = MPNNLayer(chan_hidden, chan_out)
        
        self.linear = Linear(chan_out, 1)
    
    def forward(self, x, edge_index, batch):
        
        x = self.layer1( x, edge_index)
        x = self.layer2( x, edge_index)
        x = self.layer3( x, edge_index)
        
        x = global_mean_pool(x, batch)
        
        x = self.linear(x)
        
        return x



"""
Data:
x : Knoten-Features; Tensor (Anzahl Knoten, Anzahl Feature pro Knoten)
    zB (3,2) -> [[A,B], [C,D], [E,F]]

edge_index : Kanten im Graph; Tensor (2, Kanten)
             zB (2,4) -> [[0, 1, 2, 3], [3, 3, 0, 2]]
             
             0 - 3
             | / |
             2   1
             
egde_attr : optionale Kanten Features
batch : optionaler Batch Index

"""

# Knoten-Features: 5 Knoten, jeder mit 3 Features
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=torch.float)

# Kanten-Index (diese Verbindungen repräsentieren Kanten zwischen den Knoten)
edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

# Batch-Indizes (wir haben nur einen Graphen hier, daher sind alle im selben Batch)
batch = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)

# Erstelle das Graph-Objekt
data = Data(x=x, edge_index=edge_index, batch=batch)


# Hyperparameter
in_channels = 3   # Anzahl der Features pro Knoten
hidden_channels = 6  # Verborgene Kanäle
out_channels = 4   # Ausgangskanäle

# Modell und Optimierer
model = MPNN(in_channels, hidden_channels, out_channels)
optimizer = Adam(model.parameters(), lr=0.01)

# Beispiel-Training
model.train()
for epoch in range(100):  # 100 Epochen
    optimizer.zero_grad()  # Gradienten zurücksetzen
    out = model(data.x, data.edge_index, data.batch)  # Modellvorhersage
    loss = F.mse_loss(out, torch.tensor([1.0]))  # Verlust (hier als Beispiel MSE)
    loss.backward()  # Backpropagation
    optimizer.step()  # Optimierung
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
