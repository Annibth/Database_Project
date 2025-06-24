from typing import Tuple, Any, Dict, Set

import torch
import torch.nn as nn




# A tuple representing a join action, e.g., ('table1', 'table2')
JoinAction = Tuple[Any, Any]
# A frozenset representing the current state of joined and unjoined tables
QueryState = frozenset

class TreeTransformerQNetwork(nn.Module):
    """
    A Transformer-based Q-Network for join order selection.
    It processes the set of current relations (tables or sub-plans)
    and a proposed join action to predict a Q-value.
    """
    def __init__(self, num_tables: int, embedding_dim: int = 128, nhead: int = 4, num_layers: int = 2):
        super(TreeTransformerQNetwork, self).__init__()
        self.embedding_dim = embedding_dim

        # --- Embedding Layers ---
        self.table_embeddings = nn.Embedding(num_tables, embedding_dim)
        # A special embedding for the proposed action
        self.action_type_embedding = nn.Embedding(1, embedding_dim)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=nhead, dim_feedforward=embedding_dim * 4,
            dropout=0.1, activation='relu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Output Layer ---
        # Takes the processed representation of the entire state+action
        # and outputs a single Q-value.
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )
        
    def _embed_plan_node(self, node: Any, table_to_idx: Dict[str, int], device: torch.device) -> torch.Tensor:
        """
        Create a base embedding for a plan node. If it's a sub-tree,
        it averages the embeddings of its base tables. This provides the
        initial representation before the transformer processes it.
        """
        if isinstance(node, str): # Base table
            table_idx = torch.tensor([table_to_idx[node]], dtype=torch.long, device=device)
            return self.table_embeddings(table_idx)
        elif isinstance(node, tuple): # Already joined subtree
            base_tables = self._get_base_tables_from_node(node)
            embeds = []
            for table_name in base_tables:
                 table_idx = torch.tensor([table_to_idx[table_name]], dtype=torch.long, device=device)
                 embeds.append(self.table_embeddings(table_idx))
            return torch.mean(torch.cat(embeds, dim=0), dim=0, keepdim=True)
        else:
            raise TypeError(f"Unknown node type in plan: {type(node)}")
    
    def _get_base_tables_from_node(self, node: Any) -> Set[str]:
        """Helper to recursively find all base tables in a nested tuple."""
        if isinstance(node, str):
            return {node}
        elif isinstance(node, tuple):
            return set.union(*(self._get_base_tables_from_node(child) for child in node))
        return set()

    def forward(self, state: QueryState, action: JoinAction, table_to_idx: Dict[str, int]) -> torch.Tensor:
        """
        Processes a single state-action pair using the transformer.
        """
        device = self.table_embeddings.weight.device

        # 1. Create initial embeddings for all nodes in the current state.
        state_node_embeds = [self._embed_plan_node(node, table_to_idx, device) for node in state]
        
        # 2. Create an embedding for the action. The action joins two nodes.
        # We represent the action by averaging the initial embeddings of the two nodes it joins.
        action_part1_embed = self._embed_plan_node(action[0], table_to_idx, device)
        action_part2_embed = self._embed_plan_node(action[1], table_to_idx, device)
        action_embed = (action_part1_embed + action_part2_embed) / 2

        # 3. Form a sequence for the transformer: [node1, node2, ..., action_embedding]
        # The action embedding is treated as another token in the sequence.
        input_sequence = torch.cat(state_node_embeds + [action_embed], dim=0)
        
        # Transformer expects input of shape (N, S, E) where S is sequence length. Here N=1.
        input_sequence = input_sequence.unsqueeze(0)

        # 4. Pass the sequence through the transformer encoder.
        # This allows every node to attend to every other node and to the action.
        transformer_output = self.transformer_encoder(input_sequence)

        # 5. Pool the output sequence to get a single representation for the whole state-action pair.
        # We use the embedding of the first token (mean pooling is also an option).
        pooled_output = transformer_output[:, 0, :] 

        # 6. Predict the final Q-value.
        q_value = self.output_layer(pooled_output)
        return q_value