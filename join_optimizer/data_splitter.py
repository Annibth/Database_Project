#!/usr/bin/env python3
"""
Data splitter for creating train-test splits of query data.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
from env.query_loader import load_query_from_json


class QueryDataSplitter:
    """
    Splits query data into train and test sets.
    """
    
    def __init__(self, query_data_dir: str,relation_nr:int, type:str, train_ratio: float = 0.8, 
                 random_seed: int = 42):
        self.query_data_dir = Path(query_data_dir)
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.relation_nr = relation_nr
        self.type = type
        self.relations = set()
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
    
    def load_all_queries(self) -> List:
        """Load all queries from the data directory"""
        queries = []
        query_files = list(self.query_data_dir.glob("*.json"))
        
        print(f"Loading {len(query_files)} query files...")
        length = ""
        for query_file in query_files:
            try:
                query = load_query_from_json(str(query_file))
                self.relations.add(len(query.relations))
                if self.type == "all":
                    queries.append(query)
                elif self.type == "exact":
                    if len(query.relations) == self.relation_nr:
                        queries.append(query)
                        length = f" with {self.relation_nr} relations"
                elif self.type == "leq":
                    if len(query.relations) <= self.relation_nr:
                        queries.append(query)
                        length = f" with {self.relation_nr} or less relations "
                elif self.type == "meq":
                    if len(query.relations) >= self.relation_nr and self.relation_nr!= 17:
                        queries.append(query)
                        length = f" with {self.relation_nr} or more relations"
            except Exception as e:
                print(f"Error loading {query_file}: {e}")
       
        print(f"Successfully loaded {len(queries)} queries "+ length)
        return queries
    
    
    
    def split_queries(self, queries: List) -> Tuple[List, List]:
        """
        Split queries into train and test sets.
        
        Args:
            queries: List of all queries
            
        Returns:
            train_queries: Training queries
            test_queries: Test queries
        """
        num_queries = len(queries)
        num_train = int(num_queries * self.train_ratio)
        
        # Shuffle queries
        shuffled_indices = np.random.permutation(num_queries)
        
        # Split indices
        train_indices = shuffled_indices[:num_train]
        test_indices = shuffled_indices[num_train:]
        
        # Get train and test queries
        train_queries = [queries[i] for i in train_indices]
        test_queries = [queries[i] for i in test_indices]
        
        print(f"Split {num_queries} queries:")
        print(f"  Training: {len(train_queries)} queries ({self.train_ratio*100:.1f}%)")
        print(f"  Test: {len(test_queries)} queries ({(1-self.train_ratio)*100:.1f}%)")
        
        return train_queries, test_queries
    
    def save_splits(self, train_queries: List, test_queries: List, 
                   output_dir: str = "data_splits"):
        """
        Save train and test splits to files.
        
        Args:
            train_queries: Training queries
            test_queries: Test queries
            output_dir: Directory to save splits
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save train queries
        train_file = output_path / "train_queries.json"
        train_data = {
            "queries": [query.name for query in train_queries],
            "count": len(train_queries)
        }
        with open(train_file, 'w') as f:
            json.dump(train_data, f, indent=2)
        
        # Save test queries
        test_file = output_path / "test_queries.json"
        test_data = {
            "queries": [query.name for query in test_queries],
            "count": len(test_queries)
        }
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        # Save split info
        split_info = {
            "train_ratio": self.train_ratio,
            "random_seed": self.random_seed,
            "total_queries": len(train_queries) + len(test_queries),
            "train_count": len(train_queries),
            "test_count": len(test_queries)
        }
        
        info_file = output_path / "split_info.json"
        with open(info_file, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"Saved splits to {output_path}/")
        print(f"  Train queries: {train_file}")
        print(f"  Test queries: {test_file}")
        print(f"  Split info: {info_file}")
    
    def load_splits(self, split_dir: str = "data_splits") -> Tuple[List, List]:
        """
        Load existing train-test splits.
        
        Args:
            split_dir: Directory containing split files
            
        Returns:
            train_queries: Training queries
            test_queries: Test queries
        """
        split_path = Path(split_dir)
        
        if not split_path.exists():
            raise FileNotFoundError(f"Split directory {split_dir} not found")
        
        # Load split info
        info_file = split_path / "split_info.json"
        with open(info_file, 'r') as f:
            split_info = json.load(f)
        
        # Load train queries
        train_file = split_path / "train_queries.json"
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        
        # Load test queries
        test_file = split_path / "test_queries.json"
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        # Load actual query objects
        all_queries = self.load_all_queries()
        #fours = self.load_query_selection(relations = 4, type = "exact")
        query_dict = {query.name: query for query in all_queries}
        
        train_queries = []
        for query_name in train_data["queries"]:
            if query_name in query_dict:
                train_queries.append(query_dict[query_name])
        
        test_queries = []
        for query_name in test_data["queries"]:
            if query_name in query_dict:
                test_queries.append(query_dict[query_name])
        
        print(f"Loaded splits:")
        print(f"  Training: {len(train_queries)} queries")
        print(f"  Test: {len(test_queries)} queries")
        
        return train_queries, test_queries
    
    def analyze_splits(self, train_queries: List, test_queries: List):
        """
        Analyze the characteristics of train and test splits.
        
        Args:
            train_queries: Training queries
            test_queries: Test queries
        """
        print("\n" + "="*50)
        print("SPLIT ANALYSIS")
        print("="*50)
        
        # Analyze number of relations
        train_relations = [len(query.get_relation_names()) for query in train_queries]
        test_relations = [len(query.get_relation_names()) for query in test_queries]
        
        print(f"Number of relations per query:")
        print(f"  Train - Mean: {np.mean(train_relations):.2f}, "
              f"Std: {np.std(train_relations):.2f}, "
              f"Range: {min(train_relations)}-{max(train_relations)}")
        print(f"  Test  - Mean: {np.mean(test_relations):.2f}, "
              f"Std: {np.std(test_relations):.2f}, "
              f"Range: {min(test_relations)}-{max(test_relations)}")
        
        # Analyze query types (based on first part of name)
        train_types = {}
        test_types = {}
        
        for query in train_queries:
            query_type = query.name.split()[0] if ' ' in query.name else query.name
            train_types[query_type] = train_types.get(query_type, 0) + 1
        
        for query in test_queries:
            query_type = query.name.split()[0] if ' ' in query.name else query.name
            test_types[query_type] = test_types.get(query_type, 0) + 1
        
        print(f"\nQuery type distribution:")
        all_types = set(train_types.keys()) | set(test_types.keys())
        for query_type in sorted(all_types):
            train_count = train_types.get(query_type, 0)
            test_count = test_types.get(query_type, 0)
            print(f"  {query_type}: Train={train_count}, Test={test_count}")


def main():
    """Create and save train-test splits"""
    splitter = QueryDataSplitter("query_data", train_ratio=0.8, random_seed=42)
    
    # Load all queries
    queries = splitter.load_all_queries()
    
    # Create splits
    train_queries, test_queries = splitter.split_queries(queries)
    
    # Save splits
    splitter.save_splits(train_queries, test_queries)
    
    # Analyze splits
    splitter.analyze_splits(train_queries, test_queries)


if __name__ == "__main__":
    main() 