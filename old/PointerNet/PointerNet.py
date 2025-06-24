import torch
import torch.nn as nn
import EncoderMPNN
import DecoderLSTM 

class PointerNetwork(nn.Module):
    
    def __init__(self,input_dimension, enc_hidden_dimension, enc_out_dimension):
        super().__init__()
        self.encoder = EncoderMPNN.MPNN(input_dimension, enc_hidden_dimension, enc_out_dimension)
        self.decoder = DecoderLSTM.LSTM(enc_out_dimension)
    
    def forward(self, nodes, edges, steps):
        encoded_embeddings = self.encoder(nodes, edges)
        sequence = self.decoder(encoded_embeddings, steps)
        return sequence
    

num_nodes = 4
node_feature_dim = 3
enc_hidden_dim = 10
hidden_dim = 16
max_steps = num_nodes

x = torch.randn(num_nodes, node_feature_dim)  # zufällige Knotenfeatures
edge_index = torch.tensor([
    [0, 1, 2, 3, 0, 1],
    [1, 0, 3, 2, 2, 3]
], dtype=torch.long)  # Beispiel Kanten (bidirektional)

model = PointerNetwork(node_feature_dim, enc_hidden_dim, hidden_dim)

pointers = model(x, edge_index, max_steps)
print("Sequenz der Knotenindizes:", pointers)