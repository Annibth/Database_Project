import numpy as np 
import math 
from typing import List, Set, Dict, Tuple
from env.query_loader import Query


class JoinOrderEnv:

    def __init__(self, query: Query, reward_mode: str = "cardinality", 
                 max_relations: int = 20, max_episode_length: int = None,
                 teacher_weight: float = 3.0) -> None:
        self.query = query
        self.relation_names = query.get_relation_names()
        self.num_relations = len(self.relation_names)
        self.reward_mode = reward_mode
        self.max_relations = max_relations
        self.teacher_weight = teacher_weight
        # Set episode length to actual number of relations + some buffer for exploration
        self.max_episode_length = max_episode_length if max_episode_length else self.num_relations + 2
        
        # Build join graph for feasibility checking
        self.join_graph = self._build_join_graph()
        self.current_cardinality = 0
        self.prev_cardinality = 0
        self.step_count = 0  # Track episode steps

        self.reset()
    
    def _build_join_graph(self) -> Dict[str, Set[str]]:
        """Build adjacency list representation of join graph based on sizes"""
        graph = {rel: set() for rel in self.relation_names}
        
        # Build graph from sizes (which contains all feasible join combinations)
        for size_info in self.query.sizes:
            relations = size_info["relations"]
            if len(relations) == 2:
                # Direct join between two relations
                rel1, rel2 = relations
                graph[rel1].add(rel2)
                graph[rel2].add(rel1)
            elif len(relations) > 2:
                # Multi-way join - add edges between all pairs
                for i in range(len(relations)):
                    for j in range(i + 1, len(relations)):
                        rel1, rel2 = relations[i], relations[j]
                        graph[rel1].add(rel2)
                        graph[rel2].add(rel1)
        
        return graph
    
    def _get_join_cardinality(self, relations: List[str]) -> int:
        """Get cardinality for a specific set of joined relations"""
        relations = sorted(relations)
        for size_info in self.query.sizes:
            if sorted(size_info["relations"]) == relations:
                return size_info["cardinality"]
        return 0
    
    def _is_join_feasible(self, rel1: str, rel2: str) -> bool:
        """Check if two relations can be joined based on join graph"""
        return rel2 in self.join_graph[rel1]
    
    def _get_connected_components(self, relations: Set[str]) -> List[Set[str]]:
        """Get connected components of a set of relations"""
        if not relations:
            return []
        
        components = []
        visited = set()
        
        for rel in relations:
            if rel in visited:
                continue
                
            component = set()
            stack = [rel]
            
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                    
                visited.add(current)
                component.add(current)
                
                # Add neighbors that are in the relations set
                for neighbor in self.join_graph[current]:
                    if (neighbor in relations and 
                            neighbor not in visited):
                        stack.append(neighbor)
            
            components.append(component)
        
        return components
    
    def reset(self):
        """Reset the environment to initial state"""
        self.joined = set()
        self.available = set(self.relation_names)
        self.history = []
        self.current_cardinality = 0
        self.prev_cardinality = 0
        self.done = False
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        """Get current state representation focused on join order optimization"""
        # 1. Binary vector indicating which relations are joined (padded to max_relations)
        joined_vector = np.zeros(self.max_relations, dtype=np.float32)
        for i, rel in enumerate(self.relation_names):
            if i < self.max_relations:
                joined_vector[i] = 1.0 if rel in self.joined else 0.0
        
        # 2. Current cardinality (normalized log scale)
        if self.current_cardinality > 0:
            max_cardinality = max(r["cardinality"] for r in self.query.relations)
            normalized_cardinality = math.log(self.current_cardinality) / math.log(max_cardinality)
        else:
            normalized_cardinality = 0.0
        cardinality_features = np.array([normalized_cardinality], dtype=np.float32)
        
        # 3. Table features: cardinality, connectivity, and join status for each table
        table_features = np.zeros(self.max_relations * 4, dtype=np.float32)
        for i, rel in enumerate(self.relation_names):
            if i < self.max_relations:
                # Feature 1: Normalized cardinality of this table
                for rel_info in self.query.relations:
                    if rel_info["name"] == rel:
                        max_card = max(r["cardinality"] for r in self.query.relations)
                        table_features[i * 4] = rel_info["cardinality"] / max_card
                        break
                
                # Feature 2: Connectivity (number of possible joins)
                if rel in self.join_graph:
                    table_features[i * 4 + 1] = len(self.join_graph[rel]) / self.num_relations
                
                # Feature 3: Join position (when this table was joined, 0 if not joined)
                if rel in self.joined:
                    join_position = self.history.index(rel) + 1
                    table_features[i * 4 + 2] = join_position / self.num_relations
                else:
                    table_features[i * 4 + 2] = 0.0
                
                # Feature 4: Can this table be joined now?
                table_features[i * 4 + 3] = 1.0 if i in self.get_legal_actions() else 0.0
        
        # 4. Progress features
        progress_features = np.array([
            len(self.joined) / self.num_relations,  # Completion progress
            self.step_count / self.max_episode_length,  # Time progress
            len(self.joined) / max(1, self.step_count)  # Efficiency ratio
        ], dtype=np.float32)
        
        # 5. Join order history (last 3 joins encoded as one-hot)
        history_features = np.zeros(self.max_relations, dtype=np.float32)
        for rel in self.history[-3:]:  # Last 3 joins
            if rel in self.relation_names:
                idx = self.relation_names.index(rel)
                if idx < self.max_relations:
                    history_features[idx] = 1.0
        
        # 6. Optimal order guidance (if available)
        optimal_features = np.zeros(self.max_relations, dtype=np.float32)
        if hasattr(self.query, 'left_deep_tree_min_order') and self.query.left_deep_tree_min_order:
            optimal_order = self._parse_optimal_order()
            for i, rel in enumerate(optimal_order):
                if rel in self.relation_names:
                    idx = self.relation_names.index(rel)
                    if idx < self.max_relations:
                        # Weight by position in optimal order
                        optimal_features[idx] = (len(optimal_order) - i) / len(optimal_order)
        
        # Concatenate all features
        state = np.concatenate([
            joined_vector,           # max_relations features
            cardinality_features,    # 1 feature
            table_features,          # max_relations * 4 features
            progress_features,       # 3 features
            history_features,        # max_relations features
            optimal_features         # max_relations features
        ])
        
        return state
    
    def get_legal_actions(self) -> List[int]:
        """Get list of legal action indices"""
        legal_actions = []
        
        if len(self.joined) == 0:
            # First action: can join any relation
            return list(range(self.num_relations))
        
        # Get connected components of joined relations
        components = self._get_connected_components(self.joined)
        
        # For each available relation, check if it can be joined
        for i, rel in enumerate(self.relation_names):
            if rel in self.joined:
                continue
                
            # Check if this relation can be joined to any component
            can_join = False
            for component in components:
                for joined_rel in component:
                    if self._is_join_feasible(rel, joined_rel):
                        can_join = True
                        break
                if can_join:
                    break
            
            if can_join:
                legal_actions.append(i)
        
        return legal_actions
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        self.step_count += 1
        
        # If all relations are already joined, episode is complete
        if len(self.joined) == self.num_relations:
            self.done = True
            return self._get_state(), self._calculate_reward(), True, {"complete": True}
        
        # Check if episode should end due to max length
        if self.step_count >= self.max_episode_length:
            self.done = True
            return self._get_state(), self._calculate_reward(), True, {"timeout": True}
        
        # Validate action
        legal_actions = self.get_legal_actions()
        # If there are no legal actions, end the episode gracefully
        if not legal_actions:
            self.done = True
            return self._get_state(), self._calculate_reward(), True, {"complete": True, "note": "No legal actions remaining"}
        if action not in legal_actions:
            self.done = True
            return self._get_state(), -1000.0, True, {"error": "Invalid action"}
        
        # Get the relation to join
        relation_to_join = self.relation_names[action]
        
        # Update state
        self.joined.add(relation_to_join)
        self.available.remove(relation_to_join)
        self.history.append(relation_to_join)
        
        # Calculate new cardinality
        if len(self.joined) == 1:
            # First relation: use its individual cardinality
            for rel_info in self.query.relations:
                if rel_info["name"] == relation_to_join:
                    self.prev_cardinality = self.current_cardinality
                    self.current_cardinality = rel_info["cardinality"]
                    break
        else:
            # Find the cardinality for the current set of joined relations
            new_cardinality = self._get_join_cardinality(list(self.joined))
            if new_cardinality > 0:
                self.prev_cardinality = self.current_cardinality
                self.current_cardinality = new_cardinality
        
        # If all relations are joined after this action, terminate episode
        if len(self.joined) == self.num_relations:
            self.done = True
            return self._get_state(), self._calculate_reward(), True, {
                "joined_relation": relation_to_join,
                "current_cardinality": self.current_cardinality,
                "num_joined": len(self.joined),
                "step_count": self.step_count,
                "complete": True
            }
        
        # Episode continues until completion or max_episode_length is reached
        return self._get_state(), self._calculate_reward(), False, {
            "joined_relation": relation_to_join,
            "current_cardinality": self.current_cardinality,
            "num_joined": len(self.joined),
            "step_count": self.step_count
        }
    
    def _calculate_reward(self) -> float:
        """Calculate meaningful reward based on join order quality and progress"""
        if self.reward_mode == "cardinality":
            # Reward structure that guides learning toward optimal join orders
            
            # 1. Progress reward: encourage completing the join order
            progress_reward = len(self.joined) / self.num_relations * 10.0
            
            # 2. Cardinality penalty: penalize high intermediate cardinalities
            if self.current_cardinality > 0:
                # Normalize cardinality penalty based on query size
                max_possible_cardinality = max(r["cardinality"] for r in self.query.relations) * 10
                cardinality_penalty = -math.log(self.current_cardinality) / math.log(max_possible_cardinality) * 5.0
            else:
                cardinality_penalty = 0.0
            
            # 3. Efficiency bonus: reward for good join order choices
            efficiency_bonus = 0.0
            if len(self.joined) > 1:
                # Check if current join order is following a good pattern
                # Reward for joining smaller tables first
                current_relation = self.history[-1] if self.history else None
                if current_relation:
                    for rel_info in self.query.relations:
                        if rel_info["name"] == current_relation:
                            # Reward for joining smaller tables
                            avg_cardinality = sum(r["cardinality"] for r in self.query.relations) / len(self.query.relations)
                            if rel_info["cardinality"] < avg_cardinality:
                                efficiency_bonus = 2.0
                            break

            # 3b. Teacher bonus: per-step guidance towards optimal order (if available)
            teacher_bonus = 0.0
            optimal_order_for_step = []
            if hasattr(self.query, 'left_deep_tree_min_order') and self.query.left_deep_tree_min_order:
                optimal_order_for_step = self._parse_optimal_order()
                if optimal_order_for_step and self.history:
                    # If the last joined relation matches the next optimal relation, give a small bonus
                    step_index = len(self.history) - 1
                    if step_index < len(optimal_order_for_step):
                        if self.history[-1] == optimal_order_for_step[step_index]:
                            teacher_bonus = self.teacher_weight
            
            # 4. Completion bonus: large reward for completing the join order
            completion_bonus = 0.0
            if len(self.joined) == self.num_relations:
                completion_bonus = 40.0
                
                # Additional bonus if the final join order is close to optimal
                if hasattr(self.query, 'left_deep_tree_min_order') and self.query.left_deep_tree_min_order:
                    optimal_order = self._parse_optimal_order()
                    if optimal_order:
                        accuracy = self._calculate_order_accuracy(optimal_order, self.history)
                        completion_bonus += accuracy * 80.0  # Up to 80 additional points for perfect accuracy
            
            # 5. Step penalty: small penalty for each step to encourage efficiency
            step_penalty = -0.1 * self.step_count
            
            total_reward = (
                progress_reward
                + cardinality_penalty
                + efficiency_bonus
                + teacher_bonus
                + completion_bonus
                + step_penalty
            )
            
            return total_reward
        
        # Fallback reward mode
        return 0.0
    
    def _parse_optimal_order(self) -> List[str]:
        """Parse the optimal join order from ground truth"""
        if not hasattr(self.query, 'left_deep_tree_min_order') or not self.query.left_deep_tree_min_order:
            return []
        
        order_str = self.query.left_deep_tree_min_order
        # Remove parentheses and 'join' keywords
        cleaned = order_str.replace('(', '').replace(')', '').replace(' join ', ' ')
        table_names = cleaned.split()
        
        # Extract unique tables in order
        seen = set()
        unique_tables = []
        for table in table_names:
            if table not in seen:
                seen.add(table)
                unique_tables.append(table)
        
        return unique_tables
    
    def _calculate_order_accuracy(self, optimal_order: List[str], agent_order: List[str]) -> float:
        """Calculate accuracy of join order prediction"""
        if not optimal_order or not agent_order:
            return 0.0
        
        # Calculate position-based accuracy
        correct_positions = 0
        min_length = min(len(optimal_order), len(agent_order))
        
        for i in range(min_length):
            if i < len(optimal_order) and i < len(agent_order):
                if optimal_order[i] == agent_order[i]:
                    correct_positions += 1
        
        return correct_positions / len(optimal_order) if optimal_order else 0.0
    
    def get_action_mask(self) -> np.ndarray:
        """Get binary mask for valid actions (padded to max_relations)"""
        legal_actions = self.get_legal_actions()
        mask = np.zeros(self.max_relations, dtype=np.float32)
        for action in legal_actions:
            if action < self.max_relations:
                mask[action] = 1.0
        return mask