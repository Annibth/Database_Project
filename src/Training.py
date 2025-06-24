import os
import getpass
import random
import torch
import itertools
from SQLParser import SQLParser
from RewardExecutor import PostgresExecutor
from JOSAgent import JoinOptimizerDQN
import QueryBuilder




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

# A frozenset representing the current state of joined and unjoined tables
QueryState = frozenset

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
            optimized_query = QueryBuilder.build_query_from_join_order(join_order, original_query, TPCH_JOIN_KEYS)
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