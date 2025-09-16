#!/usr/bin/env python3
"""
Evaluate trained PPO agent's join orders against ground truth
Compare join order accuracy, cost ratios, and confusion statistics
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Set
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.join_env import JoinOrderEnv
from env.query_loader import Query
from data_splitter import QueryDataSplitter
from agent.ppo import PPOAgent


class JoinOrderEvaluator:
    """Evaluate join order accuracy against ground truth"""
    
    def __init__(self, model_path: str, query_data_dir: str):
        self.model_path = model_path
        self.query_data_dir = Path(query_data_dir)
        
        # Load agent
        self.agent = self._load_agent()
        
        # Load test queries
        self.test_queries = self._load_test_queries()
        
        # Results storage
        self.results = {
            'join_order_accuracy': [],
            'cost_ratios': [],
            'confusion_stats': [],
            'query_details': [],
            'ground_truth_orders': [],
            'agent_orders': []
        }
    
    def _load_agent(self) -> PPOAgent:
        """Load the trained PPO agent"""
        print("Loading trained agent...")
        # Use the same default dims as training (state 144, action 20)
        agent = PPOAgent(state_dim=144, action_dim=20)
        
        # Load trained weights
        agent.load(self.model_path)
        print(f"Agent loaded from {self.model_path}")
        return agent
    
    def _load_test_queries(self) -> List[Query]:
        """Load test queries with ground truth information"""
        print("Loading test queries...")
        # Default to using existing saved splits
        splitter = QueryDataSplitter(self.query_data_dir, relation_nr=0, type="all")
        _, test_queries = splitter.load_splits()
        
        # Filter queries that have ground truth
        queries_with_ground_truth = []
        for query in test_queries:
            if hasattr(query, 'left_deep_tree_min_order') and query.left_deep_tree_min_order:
                queries_with_ground_truth.append(query)
        
        print(f"Loaded {len(queries_with_ground_truth)} test queries with ground truth")
        return queries_with_ground_truth
    
    def _parse_ground_truth_order(self, query: Query) -> List[str]:
        """Parse ground truth join order from string format"""
        if not hasattr(query, 'left_deep_tree_min_order') or not query.left_deep_tree_min_order:
            return []
        
        order_str = query.left_deep_tree_min_order
        # Remove parentheses and 'join' keywords
        cleaned = order_str.replace('(', '').replace(')', '').replace(' join ', ' ')
        table_names = cleaned.split()
        
        # Extract unique tables in order (handle duplicates)
        seen = set()
        unique_tables = []
        for table in table_names:
            if table not in seen:
                seen.add(table)
                unique_tables.append(table)
        
        return unique_tables
    
    def _get_agent_join_order(self, query: Query) -> List[str]:
        """Get join order from trained agent"""
        env = JoinOrderEnv(query, max_relations=20, max_episode_length=17)
        state = env.reset()
        join_order = []
        
        # Run episode to get join order
        for step in range(17):
            action_mask = env.get_action_mask()
            action, _, _ = self.agent.actor_critic.get_action(state, action_mask, deterministic=True)
            
            state, reward, done, info = env.step(action)
            
            # Record the joined relation
            if 'joined_relation' in info:
                join_order.append(info['joined_relation'])
            
            if done:
                break
        
        return join_order
    
    def _calculate_join_order_cost(self, query: Query, join_order: List[str]) -> float:
        """Calculate the cost of a specific join order"""
        if not join_order:
            return float('inf')
        
        # Find the cardinality for this join order
        for size_info in query.sizes:
            if size_info["relations"] == join_order:
                return size_info["cardinality"]
        
        # If not found, calculate cumulative cost
        total_cost = 0
        current_relations = []
        
        for relation in join_order:
            current_relations.append(relation)
            # Find cardinality for current set (order-insensitive)
            joined_sorted = sorted(current_relations)
            for size_info in query.sizes:
                if sorted(size_info["relations"]) == joined_sorted:
                    total_cost = size_info["cardinality"]
                    break
        
        return total_cost
    
    def _calculate_join_order_accuracy(self, ground_truth: List[str], agent_order: List[str]) -> float:
        """Calculate accuracy of join order prediction"""
        if not ground_truth or not agent_order:
            return 0.0
        
        # Calculate position-based accuracy
        correct_positions = 0
        min_length = min(len(ground_truth), len(agent_order))
        
        for i in range(min_length):
            if i < len(ground_truth) and i < len(agent_order):
                if ground_truth[i] == agent_order[i]:
                    correct_positions += 1
        
        return correct_positions / len(ground_truth) if ground_truth else 0.0
    
    def _calculate_confusion_stats(self, ground_truth: List[str], agent_order: List[str]) -> Dict:
        """Calculate confusion statistics for join order prediction"""
        if not ground_truth or not agent_order:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'exact_match': 0.0,
                'partial_match': 0.0
            }
        
        # Convert to sets for set-based metrics
        gt_set = set(ground_truth)
        agent_set = set(agent_order)
        
        # Calculate precision, recall, F1
        intersection = gt_set.intersection(agent_set)
        precision = len(intersection) / len(agent_set) if agent_set else 0.0
        recall = len(intersection) / len(gt_set) if gt_set else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Exact match (perfect order)
        exact_match = 1.0 if ground_truth == agent_order else 0.0
        
        # Partial match (same tables, different order)
        partial_match = 1.0 if gt_set == agent_set else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'exact_match': exact_match,
            'partial_match': partial_match
        }
    
    def evaluate_single_query(self, query: Query) -> Dict:
        """Evaluate a single query"""
        # Get ground truth order
        ground_truth_order = self._parse_ground_truth_order(query)
        if not ground_truth_order:
            return None
        
        # Get agent's join order
        agent_order = self._get_agent_join_order(query)
        
        # Calculate costs
        gt_cost = self._calculate_join_order_cost(query, ground_truth_order)
        agent_cost = self._calculate_join_order_cost(query, agent_order)
        
        # Calculate accuracy metrics
        accuracy = self._calculate_join_order_accuracy(ground_truth_order, agent_order)
        confusion_stats = self._calculate_confusion_stats(ground_truth_order, agent_order)
        
        # Calculate cost ratio
        cost_ratio = agent_cost / gt_cost if gt_cost > 0 else float('inf')
        
        return {
            'query_name': query.name,
            'ground_truth_order': ground_truth_order,
            'agent_order': agent_order,
            'ground_truth_cost': gt_cost,
            'agent_cost': agent_cost,
            'cost_ratio': cost_ratio,
            'join_order_accuracy': accuracy,
            'confusion_stats': confusion_stats
        }
    
    def evaluate_all_queries(self) -> Dict:
        """Evaluate all test queries"""
        print("Evaluating join orders against ground truth...")
        
        for query in tqdm(self.test_queries, desc="Evaluating queries"):
            result = self.evaluate_single_query(query)
            if result:
                self.results['query_details'].append(result)
                self.results['join_order_accuracy'].append(result['join_order_accuracy'])
                self.results['cost_ratios'].append(result['cost_ratio'])
                self.results['confusion_stats'].append(result['confusion_stats'])
                self.results['ground_truth_orders'].append(result['ground_truth_order'])
                self.results['agent_orders'].append(result['agent_order'])
        
        return self._compute_summary_statistics()
    
    def _compute_summary_statistics(self) -> Dict:
        """Compute summary statistics"""
        if not self.results['join_order_accuracy']:
            return {}
        
        # Filter out infinite/NaN cost ratios
        valid_cost_ratios = [r for r in self.results['cost_ratios'] if (isinstance(r, (int, float)) and np.isfinite(r))]
        
        # Calculate confusion statistics averages
        avg_confusion = {
            'precision': np.mean([cs['precision'] for cs in self.results['confusion_stats']]),
            'recall': np.mean([cs['recall'] for cs in self.results['confusion_stats']]),
            'f1_score': np.mean([cs['f1_score'] for cs in self.results['confusion_stats']]),
            'exact_match_rate': np.mean([cs['exact_match'] for cs in self.results['confusion_stats']]),
            'partial_match_rate': np.mean([cs['partial_match'] for cs in self.results['confusion_stats']])
        }
        
        return {
            'total_queries': len(self.results['query_details']),
            'mean_join_order_accuracy': np.mean(self.results['join_order_accuracy']),
            'std_join_order_accuracy': np.std(self.results['join_order_accuracy']),
            'mean_cost_ratio': np.mean(valid_cost_ratios) if valid_cost_ratios else float('inf'),
            'std_cost_ratio': np.std(valid_cost_ratios) if valid_cost_ratios else 0.0,
            'median_cost_ratio': np.median(valid_cost_ratios) if valid_cost_ratios else float('inf'),
            'confusion_statistics': avg_confusion,
            'perfect_matches': sum(1 for cs in self.results['confusion_stats'] if cs['exact_match'] == 1.0),
            'same_tables_different_order': sum(1 for cs in self.results['confusion_stats'] if cs['partial_match'] == 1.0 and cs['exact_match'] == 0.0)
        }
    
    def print_detailed_results(self):
        """Print detailed evaluation results"""
        summary = self._compute_summary_statistics()
        
        print("\n" + "="*60)
        print("JOIN ORDER EVALUATION RESULTS")
        print("="*60)
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"Total queries evaluated: {summary['total_queries']}")
        print(f"Mean join order accuracy: {summary['mean_join_order_accuracy']:.3f} ± {summary['std_join_order_accuracy']:.3f}")
        print(f"Mean cost ratio: {summary['mean_cost_ratio']:.3f} ± {summary['std_cost_ratio']:.3f}")
        print(f"Median cost ratio: {summary['median_cost_ratio']:.3f}")
        
        print(f"\n🎯 CONFUSION STATISTICS:")
        conf = summary['confusion_statistics']
        print(f"Precision: {conf['precision']:.3f}")
        print(f"Recall: {conf['recall']:.3f}")
        print(f"F1 Score: {conf['f1_score']:.3f}")
        print(f"Exact match rate: {conf['exact_match_rate']:.3f}")
        print(f"Partial match rate: {conf['partial_match_rate']:.3f}")
        
        print(f"\n🏆 PERFORMANCE BREAKDOWN:")
        print(f"Perfect matches: {summary['perfect_matches']} ({summary['perfect_matches']/summary['total_queries']*100:.1f}%)")
        print(f"Same tables, different order: {summary['same_tables_different_order']} ({summary['same_tables_different_order']/summary['total_queries']*100:.1f}%)")
        
        # Show some examples
        print(f"\n📝 EXAMPLE COMPARISONS:")
        for i, result in enumerate(self.results['query_details'][:5]):
            print(f"\nQuery {i+1}: {result['query_name']}")
            print(f"  Ground Truth: {' → '.join(result['ground_truth_order'])}")
            print(f"  Agent Order:  {' → '.join(result['agent_order'])}")
            print(f"  Accuracy: {result['join_order_accuracy']:.3f}, Cost Ratio: {result['cost_ratio']:.3f}")
    
    def save_results(self, output_dir: str = "evaluation_results"):
        """Save detailed results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save summary statistics
        summary = self._compute_summary_statistics()
        with open(output_path / "summary_statistics.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        with open(output_path / "detailed_results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Create visualization
        self._create_visualizations(output_path)
        
        print(f"\nResults saved to {output_path}")
    
    def _create_visualizations(self, output_path: Path):
        """Create visualization plots"""
        # 1. Join Order Accuracy Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.results['join_order_accuracy'], bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(self.results['join_order_accuracy']), color='red', linestyle='--', label='Mean')
        plt.title('Distribution of Join Order Accuracy')
        plt.xlabel('Accuracy')
        plt.ylabel('Number of Queries')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / "join_order_accuracy_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Cost Ratio Distribution
        valid_ratios = [r for r in self.results['cost_ratios'] if r != float('inf')]
        if valid_ratios:
            plt.figure(figsize=(10, 6))
            plt.hist(valid_ratios, bins=20, alpha=0.7, edgecolor='black')
            plt.axvline(np.mean(valid_ratios), color='red', linestyle='--', label='Mean')
            plt.axvline(1.0, color='green', linestyle='--', label='Optimal (1.0)')
            plt.title('Distribution of Cost Ratios (Agent Cost / Ground Truth Cost)')
            plt.xlabel('Cost Ratio')
            plt.ylabel('Number of Queries')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(output_path / "cost_ratio_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Confusion Statistics
        conf_stats = self.results['confusion_stats']
        metrics = ['precision', 'recall', 'f1_score', 'exact_match', 'partial_match']
        values = [np.mean([cs[m] for cs in conf_stats]) for m in metrics]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, alpha=0.7, color=['blue', 'green', 'red', 'orange', 'purple'])
        plt.title('Average Confusion Statistics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / "confusion_statistics.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main evaluation function"""
    # Check if model exists
    model_path = "models/best_model.pt"
    if not Path(model_path).exists():
        fallback = "models/final_model.pt"
        if Path(fallback).exists():
            print(f"Warning: {model_path} not found, falling back to {fallback}")
            model_path = fallback
        else:
            print(f"Error: Model file {model_path} not found and no {fallback} present!")
            print("Please train the agent first or specify the correct model path.")
            return
    
    # Initialize evaluator
    evaluator = JoinOrderEvaluator(model_path, "query_data")
    
    # Run evaluation
    summary = evaluator.evaluate_all_queries()
    
    # Print results
    evaluator.print_detailed_results()
    
    # Save results
    evaluator.save_results()
    
    print("\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main() 