import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from env.query_loader import Query
from env.join_env import JoinOrderEnv
from agent.ppo import PPOAgent


class GroundTruthEvaluator:
    """Evaluate trained agent against ground truth join orders"""
    
    def __init__(self, model_path: str, query_data_dir: str):
        self.model_path = model_path
        self.query_data_dir = Path(query_data_dir)
        
        # Load trained agent
        self.agent = self._load_agent()
        
        # Load test queries
        self.test_queries = self._load_test_queries()
    
    def _load_agent(self) -> PPOAgent:
        """Load the trained agent"""
        # Initialize agent with training dimensions (state 144, action 20)
        agent = PPOAgent(state_dim=144, action_dim=20)
        agent.load(self.model_path)
        return agent
    
    def _load_test_queries(self) -> List[Query]:
        """Load test queries with ground truth information"""
        queries = []
        query_files = list(self.query_data_dir.glob("*.json"))
        
        for file_path in query_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Only include queries with ground truth
                if "left deep tree min order" in data:
                    query = Query(
                        name=data["name"],
                        relations=data["relations"],
                        joins=data["joins"],
                        sizes=data["sizes"],
                        query=data["query"],
                        join_columns=data.get("join columns", []),
                        join_expressions=data.get("join expressions", []),
                        unary_columns=data.get("unary columns", []),
                        left_deep_tree_min_cost=data.get("left deep tree min cost", "0"),
                        bushy_deep_tree_min_cost=data.get("bushy deep tree min cost", "0"),
                        left_deep_tree_min_order=data.get("left deep tree min order", None)
                    )
                    queries.append(query)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        return queries
    
    def _parse_ground_truth_order(self, query: Query) -> List[str]:
        """Parse the ground truth join order from the data"""
        if hasattr(query, 'left_deep_tree_min_order') and query.left_deep_tree_min_order:
            # Parse the order string (e.g., "((((it join mi) join ct) join mc) join t)")
            order_str = query.left_deep_tree_min_order
            
            # Extract table names from the nested join format
            # Remove parentheses and "join" keywords, then split
            cleaned = order_str.replace('(', '').replace(')', '').replace(' join ', ' ')
            table_names = cleaned.split()
            
            # Remove duplicates while preserving order
            seen = set()
            unique_tables = []
            for table in table_names:
                if table not in seen:
                    seen.add(table)
                    unique_tables.append(table)
            
            return unique_tables
        return []
    
    def _get_agent_join_order(self, query: Query) -> List[str]:
        """Get the join order produced by the trained agent"""
        env = JoinOrderEnv(query, max_relations=20, max_episode_length=17)
        state = env.reset()
        join_order = []
        
        while not env.done and env.step_count < 17:
            action_mask = env.get_action_mask()
            action, _, _ = self.agent.actor_critic.get_action(state, action_mask, deterministic=True)
            state, _, done, info = env.step(action)
            
            if "joined_relation" in info:
                join_order.append(info["joined_relation"])
            
            if done:
                break
        
        return join_order
    
    def _calculate_join_order_cost(self, query: Query, join_order: List[str]) -> float:
        """Calculate the cost of a specific join order"""
        if not join_order:
            return float('inf')
        
        # Build the join order step by step and calculate cumulative cost
        joined = set()
        total_cost = 0
        
        for rel in join_order:
            joined.add(rel)
            joined_list = sorted(list(joined))
            
            # Find cardinality for this join combination
            cardinality = 0
            for size_info in query.sizes:
                if sorted(size_info["relations"]) == joined_list:
                    cardinality = size_info["cardinality"]
                    break
            
            if cardinality > 0:
                total_cost += cardinality
        
        return total_cost
    
    def evaluate_single_query(self, query: Query) -> Dict:
        """Evaluate a single query"""
        try:
            # Get ground truth order
            ground_truth_order = self._parse_ground_truth_order(query)
            if not ground_truth_order:
                return {"error": "No ground truth available"}
            
            # Get agent's order
            agent_order = self._get_agent_join_order(query)
            
            # Calculate costs
            ground_truth_cost = self._calculate_join_order_cost(query, ground_truth_order)
            agent_cost = self._calculate_join_order_cost(query, agent_order)
            
            # Calculate accuracy metrics
            correct_joins = 0
            total_joins = min(len(ground_truth_order), len(agent_order))
            
            for i in range(total_joins):
                if i < len(agent_order) and agent_order[i] == ground_truth_order[i]:
                    correct_joins += 1
            
            accuracy = correct_joins / total_joins if total_joins > 0 else 0
            cost_ratio = agent_cost / ground_truth_cost if ground_truth_cost > 0 else float('inf')
            
            return {
                "query_name": query.name,
                "ground_truth_order": ground_truth_order,
                "agent_order": agent_order,
                "ground_truth_cost": ground_truth_cost,
                "agent_cost": agent_cost,
                "cost_ratio": cost_ratio,
                "accuracy": accuracy,
                "correct_joins": correct_joins,
                "total_joins": total_joins
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def evaluate_all_queries(self) -> Dict:
        """Evaluate all test queries"""
        results = []
        successful_evaluations = 0
        
        for query in self.test_queries:
            result = self.evaluate_single_query(query)
            if "error" not in result:
                results.append(result)
                successful_evaluations += 1
        
        return {
            "total_queries": len(self.test_queries),
            "successful_evaluations": successful_evaluations,
            "results": results
        }
    
    def print_summary(self, evaluation: Dict):
        """Print evaluation summary"""
        if not evaluation["results"]:
            print("No successful evaluations!")
            return
        
        results = evaluation["results"]
        
        # Calculate statistics
        accuracies = [r["accuracy"] for r in results]
        cost_ratios = [r["cost_ratio"] for r in results if r["cost_ratio"] != float('inf')]
        
        print("=" * 60)
        print("GROUND TRUTH EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total queries: {evaluation['total_queries']}")
        print(f"Successful evaluations: {evaluation['successful_evaluations']}")
        print()
        
        if accuracies:
            print(f"Join Order Accuracy:")
            print(f"  Mean: {np.mean(accuracies):.3f}")
            print(f"  Std:  {np.std(accuracies):.3f}")
            print(f"  Min:  {np.min(accuracies):.3f}")
            print(f"  Max:  {np.max(accuracies):.3f}")
            print()
        
        if cost_ratios:
            print(f"Cost Ratio (Agent Cost / Ground Truth Cost):")
            print(f"  Mean: {np.mean(cost_ratios):.3f}")
            print(f"  Std:  {np.std(cost_ratios):.3f}")
            print(f"  Min:  {np.min(cost_ratios):.3f}")
            print(f"  Max:  {np.max(cost_ratios):.3f}")
            print()
        
        # Show some examples
        print("Example Results:")
        for i, result in enumerate(results[:5]):
            print(f"  Query {result['query_name']}:")
            print(f"    Ground Truth: {' -> '.join(result['ground_truth_order'])}")
            print(f"    Agent Order:  {' -> '.join(result['agent_order'])}")
            print(f"    Accuracy: {result['accuracy']:.3f}, Cost Ratio: {result['cost_ratio']:.3f}")
            print()


def main():
    """Main evaluation function"""
    evaluator = GroundTruthEvaluator(
        model_path="models/best_model.pt",
        query_data_dir="query_data"
    )
    
    print("Evaluating agent against ground truth...")
    evaluation = evaluator.evaluate_all_queries()
    evaluator.print_summary(evaluation)


if __name__ == "__main__":
    main() 