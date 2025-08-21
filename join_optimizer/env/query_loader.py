import json
from typing import List, Dict
from pathlib import Path


class Query:

    def __init__(
        self,
        name: str,
        relations: List[Dict],
        joins: List[Dict],
        sizes: List[Dict],
        query: str,
        join_columns: List[str],
        join_expressions: List[Dict],
        unary_columns: List[str] = None,
        left_deep_tree_min_cost: str = "0",
        bushy_deep_tree_min_cost: str = "0",
        left_deep_tree_min_order: str = None
    ):
        self.name = name
        self.relations = relations
        self.joins = joins
        self.sizes = sizes
        self.query = query
        self.join_columns = join_columns
        self.join_expressions = join_expressions
        self.unary_columns = unary_columns or []
        self.left_deep_tree_min_cost = left_deep_tree_min_cost
        self.bushy_deep_tree_min_cost = bushy_deep_tree_min_cost
        self.left_deep_tree_min_order = left_deep_tree_min_order

    def get_relation_names(self) -> List[str]:
        names = []
        for d in self.relations:
            names.append(d["name"])
        return names


def load_query_from_json(path: str) -> Query:
    with open(Path(path), "r") as f:
        data = json.load(f)

    name = data.get("name", "")
    relations = data["relations"]
    joins = data["joins"]
    sizes = data["sizes"]
    query = data["query"]
    join_columns = data.get("join columns", [])
    join_expressions = data.get("join expressions", [])
    unary_columns = data.get("unary columns", [])
    
    # Handle ground truth fields with spaces in names
    left_deep_tree_min_cost = data.get("left deep tree min cost", "0")
    bushy_deep_tree_min_cost = data.get("bushy deep tree min cost", "0")
    left_deep_tree_min_order = data.get("left deep tree min order", None)

    return Query(
        name=name,
        relations=relations,
        joins=joins,
        sizes=sizes,
        query=query,
        join_columns=join_columns,
        join_expressions=join_expressions,
        unary_columns=unary_columns,
        left_deep_tree_min_cost=left_deep_tree_min_cost,
        bushy_deep_tree_min_cost=bushy_deep_tree_min_cost,
        left_deep_tree_min_order=left_deep_tree_min_order
    )
