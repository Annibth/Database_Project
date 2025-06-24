import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self,hidden_dimension):
        super().__init__()
        #gleichbleibende Anzahl der Features
        self.lstm = nn.LSTMCell(hidden_dimension,hidden_dimension)
        
        # Projektion der Knoten Features und Decoder Zustand in gemeinsamen Vektorraum
        self.W1 = nn.Linear(hidden_dimension, hidden_dimension)
        self.W2 = nn.Linear(hidden_dimension, hidden_dimension)
        # Projektion auf Skalar [nodes,1]
        self.vt = nn.Linear(hidden_dimension, 1)
        
        
    def forward(self, encoded_embeddings, steps):
        
        # Anzahl der Knoten im Graphen
        nodes = encoded_embeddings.size(0)
        # Anzahl der vertseckten Features pro Knoten
        hidden_dimension = encoded_embeddings.size(1)
        
        # initiale Zustände des LSTM
        hx = torch.zeros(1,hidden_dimension)
        cx = torch.zeros(1,hidden_dimension)
        
        # initiale Eingabe für LSTM
        input = torch.zeros(1, hidden_dimension)
        
        # Maske -> gewählte Knoten
        mask = torch.zeros(nodes, dtype = torch.bool)
        
        # Liste für gewählte Elemente 
        sequence = list()
        
        for i in range(steps):
            
            # aktualisiere LSTM-Zustände mit aktuellem Input und alten Zuständen, je [1, hidden_dimension]
            hx,cx = self.lstm(input, (hx, cx))
            
            # Score-Berechnung mit Projektion von Knotenfeatures und LSTM-Zustand, Verbunden durch addition und nichtlinearität mit tanh, vt zu Skalar
            score = self.vt(torch.tanh(self.W1(encoded_embeddings) + self.W2(hx)))  
            
            # entfernen letzter Dimension [nodes, 1]-> [nodes]
            score = score.squeeze(-1)  
            
            # bereits gewählte Elemente -> score = negativ unendlich
            score = score.masked_fill(mask, float('-inf'))

            # Scores zu Wahrscheinlichkeiten
            p = F.softmax(score, dim = 0)
            
            # Wahl des Index und speichern in Liste
            index = p.argmax()
            sequence.append(index.item())
            
            # Maske updaten
            mask[index] = True
            
            # neues Input für decoder
            input = encoded_embeddings[index].unsqueeze(0)
        
        
        return sequence