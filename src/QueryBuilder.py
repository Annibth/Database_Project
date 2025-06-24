import sqlparse
from typing import List, Tuple, Any, Dict

# A tuple representing a join action, e.g., ('table1', 'table2')
JoinAction = Tuple[Any, Any]
# A frozenset representing the current state of joined and unjoined tables
QueryState = frozenset



def build_query_from_join_order(self, join_order: List[JoinAction], original_query: str, join_keys: Dict) -> str:
    """
    Constructs a final 'SELECT *' query with an explicit, nested JOIN order.
    """
    if not join_order:
        return original_query.replace(sqlparse.parse(original_query)[0].tokens[0].value, "SELECT *")

    def get_base_tables(self, node):
        if isinstance(node, str): return {node}
        return set.union(*(get_base_tables(child) for child in node))

    def build_join_string(self, node, join_keys):
        if isinstance(node, str):
            return node
        
        left_str = build_join_string(node[0], join_keys)
        right_str = build_join_string(node[1], join_keys)
        
        left_base_tables = get_base_tables(node[0])
        right_base_tables = get_base_tables(node[1])
        
        # Find the join condition between the two sub-plans
        found_key = False
        for t1 in left_base_tables:
            for t2 in right_base_tables:
                key_pair = join_keys.get(frozenset({t1, t2}))
                if key_pair:
                    # Ensure correct table.column format
                    c1, c2 = key_pair
                    if c1.split('.')[0] not in left_base_tables: c1, c2 = c2, c1
                    
                    return f"({left_str} JOIN {right_str} ON {c1} = {c2})"
        
        # Fallback if no explicit key is found in our map (should be avoided)
        return f"({left_str} NATURAL JOIN {right_str})"

    # The final join action's result is the root of the complete join tree
    final_plan_root = join_order[-1]
    from_clause = build_join_string(final_plan_root, join_keys)
    
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