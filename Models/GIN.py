import torch
import torch.nn.functional as F                     #
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data
from torch.optim import Adam

class GIN(torch.nn.Module):
    
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        
        # First Layer
        self.conv1 = GINConv(Sequential(Linear(dim_in,dim_hidden),ReLU(),Linear(dim_hidden,dim_hidden)))
        
        # further Layers ...
        
        # Output Layer, auf Output-Dimension reduziert
        self.lin = Linear(dim_hidden,dim_out)
        
    def forward(self, x, edge_index, batch):
        
        x = self.conv1(x, edge_index) #feat durch GINConv-Layer, Verarbeitung plus Nachbarinformation
        
        # further Layers
        
        x = F.relu(x) #nichtlineare Aktivierung
        
        # Aggregieren zu Graph-Embedding (weitere Funktionen)
        x = global_mean_pool(x, batch)
        
        #Output
        x = self.lin(x)
        return x
    
    
#### Data #####

x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=torch.float)

# Kanten-Index (diese Verbindungen repräsentieren Kanten zwischen den Knoten)
edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

# Batch-Indizes (wir haben nur einen Graphen hier, daher sind alle im selben Batch)
batch = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)

# Erstelle das Graph-Objekt
data = Data(x=x, edge_index=edge_index, batch=batch)  


##### Training #####
model = GIN(dim_in = 3, dim_hidden = 6, dim_out = 2)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(100):
    #zurücksetzen
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.batch)
    loss = F.mse_loss(out, torch.tensor([1.0]))  # Verlust (hier als Beispiel MSE)
    loss.backward()  # Backpropagation
    optimizer.step()  # Optimierung
    print(f"Epoch {epoch+1}, Loss: {loss.item():4f}")