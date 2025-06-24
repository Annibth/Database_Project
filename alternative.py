# main_optimizer.py
# A production-ready scaffold for a Join Order Optimizer using Deep Q-Learning (DQN).
# This code is designed to be modular and extensible.

import os
import time
import random
import collections
import itertools
from typing import List, Tuple, Dict, Any, Set

# --- Dependency Install Instructions ---
# pip install torch psycopg2-binary sqlparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sqlparse
import psycopg2
import getpass

# --- 1. Configuration & Hyperparameters ---

# Define hyperparameters for the DQN agent
# These should be tuned based on experimentation
DQN_HYPERPARAMS = {
    "GAMMA": 0.95,  # Discount factor for future rewards
    "EPSILON_START": 0.9, # Epsilon for epsilon-greedy action selection (exploration)
    "EPSILON_END": 0.05,
    "EPSILON_DECAY": 1000,
    "TARGET_UPDATE_FREQUENCY": 20, # How often to update the target network
    "BATCH_SIZE": 64, # Number of transitions to sample from the replay buffer
    "REPLAY_BUFFER_SIZE": 10000,
    "LEARNING_RATE": 1e-4,
}

# PostgreSQL Connection Details (replace with your own)
# It's recommended to use environment variables for production
POSTGRES_CONFIG = {
    "dbname": os.environ.get("PG_DBNAME", "tpch"),
    "user": os.environ.get("PG_USER", ),
    "password": os.environ.get("PG_PASSWORD", getpass.getuser()),
    "host": os.environ.get("PG_HOST", "localhost"),
    "port": os.environ.get("PG_PORT", "5432"),
}

# TPC-H Schema Join Key Information (simplified)
# In a real system, this could be discovered from DB catalogs.
# Format: frozenset({'table1', 'table2'}): ('table1.key', 'table2.key')
TPCH_JOIN_KEYS = {
    frozenset({'customer', 'orders'}): ('c_custkey', 'o_custkey'),
    frozenset({'orders', 'lineitem'}): ('o_orderkey', 'l_orderkey'),
    frozenset({'part', 'partsupp'}): ('p_partkey', 'ps_partkey'),
    frozenset({'supplier', 'partsupp'}): ('s_suppkey', 'ps_suppkey'),
    frozenset({'nation', 'supplier'}): ('n_nationkey', 's_nationkey'),
    frozenset({'nation', 'customer'}): ('n_nationkey', 'c_nationkey'),
    frozenset({'region', 'nation'}): ('r_regionkey', 'n_nationkey'),
    frozenset({'part', 'lineitem'}): ('p_partkey', 'l_partkey'),
    frozenset({'supplier', 'lineitem'}): ('s_suppkey', 'l_suppkey'),
}

# --- 2. Core Components ---

class SQLParser:
    """
    Parses an SQL query to extract table names.
    This is a simplified parser focusing on FROM and JOIN clauses.
    """
    def get_tables(self, sql_query: str) -> List[str]:
        """Extracts table names from the FROM and JOIN clauses of an SQL query."""
        tables = set()
        parsed = sqlparse.parse(sql_query)[0]
        from_seen = False
        for token in parsed.tokens:
            if isinstance(token, sqlparse.sql.Where):
                # Stop parsing after the WHERE clause
                break
            if from_seen:
                # This is a simplification. It handles basic "table alias" but not complex subqueries.
                if token.ttype is sqlparse.tokens.Keyword and token.value.upper() not in ['AS', 'ON', 'USING']:
                    continue
                if isinstance(token, sqlparse.sql.Identifier):
                    tables.add(token.get_real_name())
                elif isinstance(token, sqlparse.sql.IdentifierList):
                    for identifier in token.get_identifiers():
                        tables.add(identifier.get_real_name())

            if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'FROM':
                from_seen = True
            # Also capture tables from JOIN clauses
            if token.ttype is sqlparse.tokens.Keyword and 'JOIN' in token.value.upper():
                # The next non-keyword, non-whitespace token should be the table
                next_token_idx = parsed.token_index(token) + 1
                while next_token_idx < len(parsed.tokens):
                    next_token = parsed.tokens[next_token_idx]
                    if next_token.is_whitespace:
                        next_token_idx += 1
                        continue
                    if isinstance(next_token, sqlparse.sql.Identifier):
                        tables.add(next_token.get_real_name())
                    break

        print(f"Parsed tables: {tables} from query.")
        return sorted(list(tables)) # Return a sorted list for consistency

class PostgresExecutor:
    """Handles executing queries against the PostgreSQL database and measuring latency."""
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self._connection = None

    def _get_connection(self):
        """Establishes or reuses a database connection."""
        if self._connection is None or self._connection.closed:
            try:
                self._connection = psycopg2.connect(**self.config)
            except psycopg2.OperationalError as e:
                print(f"Error connecting to PostgreSQL: {e}")
                raise
        return self._connection

    def execute_query(self, query: str) -> float:
        """
        Executes a given SQL query and returns the execution time in seconds.
        Returns float('inf') on error.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Use EXPLAIN ANALYZE to get the actual execution time from the planner
        explain_query = f"EXPLAIN ANALYZE {query}"
        
        try:
            start_time = time.perf_counter()
            cursor.execute(explain_query)
            result = cursor.fetchall()
            end_time = time.perf_counter()

            # The last line of EXPLAIN ANALYZE contains the execution time
            execution_time_line = [line[0] for line in result if "Execution Time" in line[0]]
            if execution_time_line:
                # Example line: "Execution Time: 123.456 ms"
                time_str = execution_time_line[0].split(":")[1].strip()
                value, unit = time_str.split()
                if unit == 'ms':
                    return float(value) / 1000.0
                else: # Assuming seconds if not ms
                    return float(value)
            else:
                # Fallback to wall-clock time if parsing fails
                print("Warning: Could not parse execution time from EXPLAIN ANALYZE. Using wall-clock time.")
                return end_time - start_time

        except psycopg2.Error as e:
            print(f"Error executing query: {e}")
            conn.rollback()
            return float('inf') # Return a very high cost for failed queries
        finally:
            cursor.close()

    def close(self):
        """Closes the database connection if it's open."""
        if self._connection and not self._connection.closed:
            self._connection.close()

# --- 3. The Neural Network (Tree-Transformer Placeholder) ---
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

# --- 4. Reinforcement Learning Agent (DQN) ---

# A named tuple to store experience transitions in the replay buffer
Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """A simple circular buffer to store experience transitions."""
    def __init__(self, capacity: int):
        self.memory = collections.deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a batch of transitions."""
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class JoinOptimizerDQN:
    """The main DQN agent that orchestrates the optimization process."""

    def __init__(self, all_tpch_tables: List[str], params: Dict, device: torch.device):
        self.params = params
        self.device = device
        
        # Vocabulary mapping for embeddings
        self.all_tables = sorted(all_tpch_tables)
        self.table_to_idx = {name: i for i, name in enumerate(self.all_tables)}
        num_tables = len(self.all_tables)

        # Initialize policy and target networks
        self.policy_net = TreeTransformerQNetwork(num_tables).to(device)
        self.target_net = TreeTransformerQNetwork(num_tables).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is only for inference

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=params["LEARNING_RATE"])
        self.replay_buffer = ReplayBuffer(params["REPLAY_BUFFER_SIZE"])
        self.steps_done = 0

    def get_possible_actions(self, state: QueryState) -> List[JoinAction]:
        """
        Given a state (a set of tables/subtrees), return all possible join actions.
        An action is joining any two elements from the set.
        """
        # Create all unique pairs of nodes in the current state
        return list(itertools.combinations(state, 2))

    def select_action(self, state: QueryState) -> JoinAction:
        """
        Selects an action using an epsilon-greedy policy.
        With probability epsilon, take a random action (explore).
        Otherwise, take the best action according to the policy network (exploit).
        """
        sample = random.random()
        eps_threshold = self.params["EPSILON_END"] + (self.params["EPSILON_START"] - self.params["EPSILON_END"]) * \
                        (1. - min(1.0, self.steps_done / self.params["EPSILON_DECAY"]))
        self.steps_done += 1
        
        possible_actions = self.get_possible_actions(state)

        if sample > eps_threshold:
            # --- Exploitation: Choose the best action from the policy network ---
            with torch.no_grad():
                q_values = [self.policy_net(state, action, self.table_to_idx) for action in possible_actions]
                # Find the index of the action with the highest Q-value
                best_action_idx = torch.cat(q_values).argmax().item()
                return possible_actions[best_action_idx]
        else:
            # --- Exploration: Choose a random action ---
            return random.choice(possible_actions)

    def optimize_model(self):
        """Performs one step of optimization on the policy network."""
        if len(self.replay_buffer) < self.params["BATCH_SIZE"]:
            return # Not enough samples yet

        transitions = self.replay_buffer.sample(self.params["BATCH_SIZE"])
        batch = Transition(*zip(*transitions))

        # --- 1. Compute Q(s_t, a) for the current batch ---
        # These are the Q-values our policy network predicted for the actions taken.
        state_action_values = []
        for state, action in zip(batch.state, batch.action):
            q_val = self.policy_net(state, action, self.table_to_idx)
            state_action_values.append(q_val)
        state_action_values = torch.cat(state_action_values)

        # --- 2. Compute V(s_{t+1}) for the next states ---
        # This is the maximum Q-value for the next state, predicted by the target network.
        # This is 0 if the state was terminal.
        next_state_values = torch.zeros(self.params["BATCH_SIZE"], device=self.device)
        non_final_next_states = [s for s in batch.next_state if not s is None]
        
        if non_final_next_states:
            # For each non-final next state, find the max Q-value of all possible actions
            max_q_values_for_next_states = []
            for next_state in non_final_next_states:
                possible_actions = self.get_possible_actions(next_state)
                q_values = [self.target_net(next_state, action, self.table_to_idx) for action in possible_actions]
                max_q_value = torch.cat(q_values).max().unsqueeze(0)
                max_q_values_for_next_states.append(max_q_value)
            
            # Create a tensor of the max Q-values
            next_state_max_qs = torch.cat(max_q_values_for_next_states)
            
            # Find the indices of the non-final states in the original batch
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
            next_state_values[non_final_mask] = next_state_max_qs.squeeze()


        # --- 3. Compute the expected Q values (Bellman equation) ---
        reward_batch = torch.cat(batch.reward)
        expected_state_action_values = (next_state_values * self.params["GAMMA"]) + reward_batch

        # --- 4. Compute Loss and Optimize ---
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()
        
    def update_target_net(self):
        """Copies weights from the policy network to the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

def build_query_from_join_order(join_order: List[JoinAction], original_query: str) -> str:
    """
    Constructs a final 'SELECT *' query with an explicit, nested JOIN order.
    """
    if not join_order:
        return original_query.replace(sqlparse.parse(original_query)[0].tokens[0].value, "SELECT *")

    def get_base_tables(node):
        if isinstance(node, str): return {node}
        return set.union(*(get_base_tables(child) for child in node))

    def build_join_string(node):
        if isinstance(node, str):
            return node
        
        left_str = build_join_string(node[0])
        right_str = build_join_string(node[1])
        
        left_base_tables = get_base_tables(node[0])
        right_base_tables = get_base_tables(node[1])
        
        # Find the join condition between the two sub-plans
        found_key = False
        for t1 in left_base_tables:
            for t2 in right_base_tables:
                key_pair = TPCH_JOIN_KEYS.get(frozenset({t1, t2}))
                if key_pair:
                    # Ensure correct table.column format
                    c1, c2 = key_pair
                    if c1.split('.')[0] not in left_base_tables: c1, c2 = c2, c1
                    
                    return f"({left_str} JOIN {right_str} ON {c1} = {c2})"
        
        # Fallback if no explicit key is found in our map (should be avoided)
        return f"({left_str} NATURAL JOIN {right_str})"

    # The final join action's result is the root of the complete join tree
    final_plan_root = join_order[-1]
    from_clause = build_join_string(final_plan_root)
    
    # Extract WHERE clause and other clauses from original query
    parsed = sqlparse.parse(original_query)[0]
    where_and_after = ""
    where_found = False
    for token in parsed.tokens:
        if isinstance(token, sqlparse.sql.Where):
            where_found = True
        if where_found:
            where_and_after += token.value
            
    final_query = f"SELECT * FROM {from_clause} {where_and_after}"
    return final_query


# --- 5. The Main Training Loop ---

def main():
    """Main function to run the training process."""
    print("--- Starting Join Order Optimizer Training ---")
    
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use a fixed list of all tables in the TPC-H schema for the vocabulary
    all_tpch_tables = list(set(itertools.chain.from_iterable(TPCH_JOIN_KEYS.keys())))
    
    parser = SQLParser()
    executor = PostgresExecutor(POSTGRES_CONFIG)
    agent = JoinOptimizerDQN(all_tpch_tables, DQN_HYPERPARAMS, device)

    # Example TPC-H based queries (replace with your query generator)
    queries = [
        "SELECT c_name, o_orderdate FROM customer JOIN orders ON c_custkey = o_custkey JOIN nation ON c_nationkey = n_nationkey WHERE n_name = 'GERMANY'",
        "SELECT s_name, p_name FROM part JOIN partsupp ON p_partkey = ps_partkey JOIN supplier ON ps_suppkey = s_suppkey WHERE p_size < 10",
        "SELECT c_name, n_name, r_name FROM region JOIN nation ON r_regionkey = n_regionkey JOIN customer ON n_nationkey = c_nationkey"
    ]
    
    num_episodes = 2
    print(f"Starting training for {num_episodes} episodes...")

    # --- Training Loop ---
    for episode in range(num_episodes):
        try:
            original_query = random.choice(queries)
            initial_tables = parser.get_tables(original_query)
            
            if len(initial_tables) < 2:
                print(f"Skipping query with < 2 tables: {original_query}")
                continue

            # --- RL Episode: Agent generates a join plan ---
            state = QueryState(initial_tables)
            join_order = []
            episode_transitions = []

            while len(state) > 1:
                print("start")
                action = agent.select_action(state)
                # Create the next state by merging the two parts of the action
                next_state_list = [node for node in state if node not in action]
                print(f"Next state list: {next_state_list}")
                # Sort tuple elements for consistency
                new_node = tuple(action)
                next_state_list.append(new_node)
                
                # Check if this is the final join
                is_done = len(next_state_list) == 1
                next_state = QueryState(next_state_list) if not is_done else None
                print(f"Next state: {next_state}, type {type(next_state)}")
                # Store the transition without the reward (we get it at the end)
                episode_transitions.append({'state': state, 'action': action, 'next_state': next_state, 'done': is_done})
                join_order.append(action)
                if next_state is not None:
                    state = next_state
                else:
                    break
                #print(len(state) > 1)
            
            # --- Execute the generated plan and get the reward ---
            print("Start building optimized query")
            optimized_query = build_query_from_join_order(join_order, original_query)
            print(f"\n[Episode {episode+1}/{num_episodes}]")
            print(f"  - Optimized Query Plan: {join_order}")
            print(f"  - Generated SQL: {optimized_query}")
            
            execution_time = executor.execute_query(optimized_query)


            # Define the reward. Lower execution time is better.
            # We use the negative of the time so the agent learns to maximize a less negative number.
            reward = -execution_time
            print(f"  - Execution Time: {execution_time:.4f}s, Reward: {reward:.4f}")
            
            # --- Store episode transitions in replay buffer with the final reward ---
            # The terminal reward is applied to all steps in the episode
            reward_tensor = torch.tensor([reward], device=device, dtype=torch.float)
            for trans in episode_transitions:
                agent.replay_buffer.push(trans['state'], trans['action'], reward_tensor, trans['next_state'], trans['done'])

            # --- Optimize the model ---
            loss = agent.optimize_model()
            if loss is not None:
                print(f"  - Loss: {loss:.5f}")

            # --- Update Target Network ---
            if episode % DQN_HYPERPARAMS["TARGET_UPDATE_FREQUENCY"] == 0:
                print("  - Updating target network...")
                agent.update_target_net()

        except Exception as e:
            print(f"An error occurred in episode {episode+1}: {e}")
            continue

    # --- Cleanup ---
    executor.close()
    print("\n--- Training Finished ---")
    
    # You would save the trained model here
    # torch.save(agent.policy_net.state_dict(), 'join_optimizer_dqn_model.pth')


if __name__ == "__main__":
    main()