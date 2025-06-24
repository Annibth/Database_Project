import collections
from typing import List, Dict, Tuple, Any
import random
import itertools


import torch
from TreeTransformer import TreeTransformerQNetwork
import torch.nn.functional as F
import torch.optim as optim

# A named tuple to store experience transitions in the replay buffer
Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
# A tuple representing a join action, e.g., ('table1', 'table2')
JoinAction = Tuple[Any, Any]
# A frozenset representing the current state of joined and unjoined tables
QueryState = frozenset

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
