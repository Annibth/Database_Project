import numpy as np 
import math 
from typing import List, Set, Dict, Tuple
from env.query_loader import Query


class JoinOrderEnv:

    def __init__(self, query: Query, reward_mode="cardinality", 
                 max_relations=20, max_episode_length=17) -> None:
        self.query = query
        self.relation_names = query.get_relation_names()
        self.num_relations = len(self.relation_names)
        self.reward_mode = reward_mode
        self.max_relations = max_relations
        self.max_episode_length = max_episode_length
        
        # Build join graph for feasibility checking
        self.join_graph = self._build_join_graph()
        self.current_cardinality = 0
        self.join_history = []
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
        self.done = False
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        """Get current state representation with enhanced position-aware features"""
        # Binary vector indicating which relations are joined
        joined_vector = np.zeros(self.max_relations, dtype=np.float32)
        for i, rel in enumerate(self.relation_names):
            if i < self.max_relations:
                joined_vector[i] = 1.0 if rel in self.joined else 0.0
        
        # Current cardinality (log-scaled)
        log_cardinality = np.array([math.log(max(1, self.current_cardinality))], dtype=np.float32)
        
        # Enhanced table features: cardinality and connectivity for each table
        table_features = np.zeros(self.max_relations * 3, dtype=np.float32)  # Increased to 3 features per table
        for i, rel in enumerate(self.relation_names):
            if i < self.max_relations:
                # Feature 1: Log cardinality
                for rel_info in self.query.relations:
                    if rel_info["name"] == rel:
                        table_features[i * 3] = math.log(max(1, rel_info["cardinality"]))
                        break
                
                # Feature 2: Connectivity (number of possible joins)
                if rel in self.join_graph:
                    table_features[i * 3 + 1] = len(self.join_graph[rel])
                
                # Feature 3: Join position (when this table was joined, 0 if not joined)
                if rel in self.joined:
                    join_position = self.history.index(rel) + 1
                    table_features[i * 3 + 2] = join_position / self.num_relations  # Normalized position
                else:
                    table_features[i * 3 + 2] = 0.0
        
        # Progress indicator (normalized episode progress)
        progress = np.array([self.step_count / self.max_episode_length], dtype=np.float32)
        
        # Join order history features (last 5 joins)
        history_features = np.zeros(5, dtype=np.float32)
        for i, rel in enumerate(self.history[-5:]):  # Last 5 joins
            if i < 5:
                # Encode the relation name as a simple hash
                rel_hash = hash(rel) % 1000 / 1000.0  # Normalized hash
                history_features[i] = rel_hash
        
        # Current join feasibility features
        feasibility_features = np.zeros(self.max_relations, dtype=np.float32)
        legal_actions = self.get_legal_actions()
        for action in legal_actions:
            if action < self.max_relations:
                feasibility_features[action] = 1.0
        
        # Concatenate all features
        state = np.concatenate([
            joined_vector,           # 20 features
            log_cardinality,         # 1 feature
            table_features,          # 60 features (20 * 3)
            progress,                # 1 feature
            history_features,        # 5 features
            feasibility_features     # 20 features
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
        
        # Check if episode should end due to max length (exactly 17 steps)
        if self.step_count >= self.max_episode_length:
            self.done = True
            return self._get_state(), 0.0, True, {"timeout": True}
        
        # If all relations are already joined, continue with no-op actions
        if len(self.joined) == self.num_relations:
            return self._get_state(), 0.0, False, {"no_op": True}
        
        # Validate action
        legal_actions = self.get_legal_actions()
        if action not in legal_actions:
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
                    self.current_cardinality = rel_info["cardinality"]
                    break
        else:
            # Find the cardinality for the current set of joined relations
            new_cardinality = self._get_join_cardinality(list(self.joined))
            if new_cardinality > 0:
                self.current_cardinality = new_cardinality
        
        # Episode continues until max_episode_length is reached
        return self._get_state(), self._calculate_reward(), False, {
            "joined_relation": relation_to_join,
            "current_cardinality": self.current_cardinality,
            "num_joined": len(self.joined),
            "step_count": self.step_count
        }
    
    def _calculate_reward(self) -> float:
        """Calculate simplified reward using only base_reward and efficiency_bonus"""
        if self.reward_mode == "cardinality":
            # Simplified reward structure using only base_reward and efficiency_bonus
            
            # Base reward: negative log cardinality (we want low cardinalities)
            # Scale down to prevent very negative rewards
            base_reward = -math.log(max(1, self.current_cardinality)) / 15.0
            
            # Efficiency bonus: reward for joining small tables first
            efficiency_bonus = 0.0
            if len(self.joined) == 1:
                # Find the joined table's cardinality
                for rel_info in self.query.relations:
                    if rel_info["name"] in self.joined:
                        # Reward for starting with small tables
                        total_cardinality = sum(r["cardinality"] for r in self.query.relations)
                        efficiency_bonus = (total_cardinality - rel_info["cardinality"]) / total_cardinality * 3.0
                        break
            
            total_reward = base_reward + efficiency_bonus
            
            return total_reward
        
        # Fallback reward mode
        return 0.0
    
    def get_action_mask(self) -> np.ndarray:
        """Get binary mask for valid actions (padded to max_relations)"""
        legal_actions = self.get_legal_actions()
        mask = np.zeros(self.max_relations, dtype=np.float32)
        for action in legal_actions:
            if action < self.max_relations:
                mask[action] = 1.0
        return mask
    
    def get_optimal_cost(self) -> float:
        """Get the optimal cost for this query"""
        # This would need to be implemented based on your optimal solution data
        # For now, return a placeholder
        return 0.0