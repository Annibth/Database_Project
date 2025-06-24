import psycopg2
import getpass
import itertools
import time
import json
import logging
from sqlparse import format as sql_format, parse, split
from sqlparse import sql, tokens
from typing import Dict, List, Any, Optional

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# It's recommended to use environment variables or a secure config file for production
# For this script, we'll derive the username automatically.
current_user = getpass.getuser()
DB_CONFIG = {
    "dbname": "tpch",
    "user": current_user,
    "password": "",  # Assumes local user has access, e.g., via 'ident' or 'trust'
    "host": "localhost",
    "port": "5432"
}

# --- File and Execution Parameters ---
INPUT_QUERIES = "data/queries/tpch_sample_queries_test.sql"
OUTPUT_FILE = "data/generated/tpch_dataset.jsonl"
MAX_ENUM = 32  # Max number of join permutations to test exhaustively

# --- TPC-H Schema Join Definitions ---
# This dictionary defines the standard join conditions between tables in the TPC-H schema.
# It is used to construct new queries with different join orders.
BASE_CONDITIONS = {
    ("lineitem", "orders"): "lineitem.l_orderkey = orders.o_orderkey",
    ("orders", "customer"): "orders.o_custkey = customer.c_custkey",
    ("customer", "nation"): "customer.c_nationkey = nation.n_nationkey",
    ("nation", "region"): "nation.n_regionkey = region.r_regionkey",
    ("supplier", "nation"): "supplier.s_nationkey = nation.n_nationkey",
    ("partsupp", "supplier"): "partsupp.ps_suppkey = supplier.s_suppkey",
    ("part", "partsupp"): "part.p_partkey = partsupp.ps_partkey",
    ("lineitem", "partsupp"): "lineitem.l_suppkey = partsupp.ps_suppkey AND lineitem.l_partkey = partsupp.ps_partkey",
    ("lineitem", "supplier"): "lineitem.l_suppkey = supplier.s_suppkey",
    ("lineitem", "part"): "lineitem.l_partkey = part.p_partkey",
}


def get_connection() -> Optional[psycopg2.extensions.connection]:
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logging.info(f"Successfully connected to PostgreSQL database '{DB_CONFIG['dbname']}' as user '{current_user}'.")
        return conn
    except psycopg2.OperationalError as e:
        logging.error("--- DATABASE CONNECTION FAILED ---")
        logging.error(f"Error: {e}")
        logging.error("Please check if the database is running and if DB_CONFIG is correct.")
        return None


def extract_tables(query: str) -> List[str]:
    """
    Parses a SQL query and extracts all unique table names from FROM and JOIN clauses.
    This version is a more robust, stateless implementation.
    """
    parsed = parse(query)[0]
    print("PARSED: ", parsed)
    tables = []
    
    # Get the list of all tokens in the parsed statement
    tokens = parsed.tokens

    # Find the indexes of all FROM and JOIN keywords
    from_or_join_indices = [
        i for i, token in enumerate(tokens)
        if token.is_keyword and token.value.upper() in ('FROM', 'JOIN')
    ]

    for i in from_or_join_indices:
        # The table name should be the next non-whitespace token after the keyword.
        next_token_index = i + 1
        while next_token_index < len(tokens) and tokens[next_token_index].is_whitespace:
            next_token_index += 1
        
        if next_token_index < len(tokens):
            table_token = tokens[next_token_index]
            # The token can be a single identifier or a list of them
            if isinstance(table_token, sql.Identifier):
                tables.append(table_token.get_real_name())
            elif isinstance(table_token, sql.IdentifierList):
                 for identifier in table_token.get_identifiers():
                    tables.append(identifier.get_real_name())

    # Return a unique list of tables, preserving insertion order.
    unique_tables = list(dict.fromkeys(tables))
    return unique_tables


def extract_where_clause(query: str) -> Optional[str]:
    """Extracts the WHERE clause from a SQL query, if one exists."""
    parsed = parse(query)[0]
    where_seen = False
    where_tokens = []
    for token in parsed.tokens:
        if where_seen:
            if token.is_keyword and token.value.upper() in ('GROUP BY', 'ORDER BY', 'LIMIT', 'HAVING', 'UNION'):
                break
            where_tokens.append(str(token))

        if token.is_keyword and token.value.upper() == 'WHERE':
            where_seen = True

    return "".join(where_tokens).strip() if where_tokens else None


def explain_analyze(conn: psycopg2.extensions.connection, sql_query: str, analyze: bool = True) -> Dict:
    """
    Runs EXPLAIN on a query and returns the plan as a dictionary.
    """
    options = ["FORMAT JSON", "BUFFERS"]
    if analyze:
        options.extend(["ANALYZE", "TIMING"])

    explain_prefix = f"EXPLAIN ({', '.join(options)}) "

    with conn.cursor() as cur:
        cur.execute("SET LOCAL join_collapse_limit = 1; SET LOCAL from_collapse_limit = 1;")
        cur.execute("SET LOCAL enable_hashjoin = on; SET LOCAL enable_mergejoin = off; SET LOCAL enable_nestloop = off;")
        cur.execute(explain_prefix + sql_query)
        plan_json = cur.fetchone()[0][0]
    return plan_json


def make_left_deep_sql(order: List[str], base_conditions: Dict, where_clause: Optional[str]) -> str:
    """
    Constructs a SQL query with a specified left-deep join order.
    """
    if not order:
        return ""

    sql = order[0]
    for i in range(1, len(order)):
        t1, t2 = order[i - 1], order[i]
        cond = base_conditions.get((t1, t2)) or base_conditions.get((t2, t1))
        if not cond:
            raise KeyError(f"Missing join condition for the pair ({t1}, {t2}) in BASE_CONDITIONS.")
        sql = f"({sql} JOIN {t2} ON {cond})"

    final_sql = f"SELECT * FROM {sql}"
    if where_clause:
        final_sql += f" WHERE {where_clause}"

    return final_sql + ";"


def enumerate_orders(conn: psycopg2.extensions.connection, tables: List[str], where_clause: Optional[str], max_tries: int) -> List[List[str]]:
    """
    Generates permutations of join orders, scores them, and returns the best N.
    """
    if len(tables) > 9: # n! for n=10 is 3.6 million, getting very slow
        logging.warning(f"Query has {len(tables)} tables. Permutation count is large. Scoring may be slow.")

    all_orders = list(itertools.permutations(tables))

    if len(all_orders) <= max_tries:
        return [list(o) for o in all_orders]

    logging.info(f"Found {len(all_orders)} permutations for {tables}, scoring to find the best {max_tries} candidates.")
    scored = []
    for order in all_orders:
        try:
            sql = make_left_deep_sql(list(order), BASE_CONDITIONS, where_clause)
            plan = explain_analyze(conn, sql, analyze=False)
            cost = plan["Plan"]["Total Cost"]
            scored.append((cost, order))
        except (KeyError, psycopg2.Error) as e:
            logging.debug(f"Could not generate plan for order {order}: {e}")
            continue
            
    scored.sort(key=lambda x: x[0])
    logging.info(f"Selected top {max_tries} join orders for analysis.")
    return [list(order) for _, order in scored[:max_tries]]

def parse_plan_to_graph(plan: Dict[str, Any]) -> Dict[str, List[Any]]:
    """
    Parses a recursive PostgreSQL EXPLAIN plan into a flat graph structure
    (nodes and edges) suitable for a Graph Neural Network.
    """
    nodes = []
    edges = []
    node_counter = 0

    def _traverse_node(node_data: Dict[str, Any], parent_id: Optional[int]):
        nonlocal node_counter
        current_id = node_counter
        node_counter += 1

        # Extract features for the current node
        features = {
            "id": current_id,
            "node_type": node_data.get("Node Type"),
            "relation_name": node_data.get("Relation Name"),
            "alias": node_data.get("Alias"),
            "group_key": node_data.get("Group Key"),
            "sort_key": node_data.get("Sort Key"),
            "join_type": node_data.get("Join Type"),
            "est_startup_cost": node_data.get("Startup Cost"),
            "est_total_cost": node_data.get("Total Cost"),
            "est_rows": node_data.get("Plan Rows"),
            "est_width": node_data.get("Plan Width"),
            "actual_startup_time": node_data.get("Actual Startup Time"),
            "actual_total_time": node_data.get("Actual Total Time"),
            "actual_rows": node_data.get("Actual Rows"),
            "actual_loops": node_data.get("Actual Loops"),
            "shared_hit_blocks": node_data.get("Shared Hit Blocks"),
            "shared_read_blocks": node_data.get("Shared Read Blocks"),
        }
        nodes.append(features)

        # Create an edge from this node to its parent
        if parent_id is not None:
            edges.append({"source": current_id, "target": parent_id})

        # Recurse for child plans
        if "Plans" in node_data:
            for child_plan in node_data["Plans"]:
                _traverse_node(child_plan, parent_id=current_id)

    # Start traversal from the root of the plan
    _traverse_node(plan["Plan"], parent_id=None)
    return {"nodes": nodes, "edges": edges}

def main():
    """Main execution function."""
    conn = get_connection()
    if not conn:
        return

    try:
        with open(INPUT_QUERIES, "r") as fin, open(OUTPUT_FILE, "w") as fout:
            content = fin.read()
            queries = [q.strip() for q in split(content) if q.strip()]
            for i, raw_sql in enumerate(queries):
                print("RAW_SQL: ", raw_sql)
                if not raw_sql.strip():
                    continue

                logging.info(f"--- Processing Query {i+1} ---")
                q = sql_format(raw_sql.strip(), reindent=True, indent_width=2)
                
                try:
                    tables = extract_tables(q)
                    where_clause = extract_where_clause(q)
                    logging.info(f"Extracted Tables: {tables}")
                    if where_clause:
                        logging.info(f"Extracted WHERE clause: {where_clause[:100]}...")

                    if len(tables) < 2:
                        logging.warning("Query has fewer than 2 tables, skipping join optimization.")
                        continue

                    # 1. Get baseline plan from PostgreSQL's default optimizer
                    pg_plan_json = explain_analyze(conn, q)
                    pg_time = pg_plan_json["Execution Time"]
                    pg_graph = parse_plan_to_graph(pg_plan_json) # Parse into GNN format

                    # 2. Find the best left-deep plan
                    all_orders = list(itertools.permutations(tables))
                    best_result = None
                    for order in all_orders:
                        try:
                            sql_o = make_left_deep_sql(list(order), BASE_CONDITIONS, where_clause)
                            plan_o_json = explain_analyze(conn, sql_o)
                            exec_t = plan_o_json["Execution Time"]

                            if best_result is None or exec_t < best_result["exec_time"]:
                                best_result = {
                                    "exec_time": exec_t,
                                    "order": list(order),
                                    "plan_json": plan_o_json, # Store the full plan
                                    "sql": sql_o
                                }
                        except (KeyError, psycopg2.Error) as e:
                            logging.warning(f"Failed to analyze order {order}: {e}")
                            conn.rollback()
                            continue
                    
                    if not best_result:
                        logging.error("Failed to find any successful optimized plan.")
                        continue

                    # 3. Parse the optimized plan into GNN format
                    optimized_graph = parse_plan_to_graph(best_result["plan_json"])
                    optimized_time = best_result['exec_time']
                    
                    # 4. Write the structured data to the output file
                    output_data = {
                        "original_query": q,
                        "tables": tables,
                        "postgres_default_plan": {
                            "execution_time_ms": pg_time,
                            "parsed_graph": pg_graph # GNN-ready graph
                        },
                        "optimized_left_deep_plan": {
                            "join_order": best_result["order"],
                            "execution_time_ms": optimized_time,
                            "parsed_graph": optimized_graph # GNN-ready graph
                        }
                    }
                    fout.write(json.dumps(output_data, indent=2) + "\n")


                    logging.info(
                        f"✅ Success for tables {tables} | "
                        f"PG Time: {pg_time:.2f} ms → Optimized Time: {best_result['exec_time']:.2f} ms"
                    )

                except (psycopg2.Error, KeyError) as e:
                    logging.error(f"An error occurred while processing query: {q}\nError: {e}")
                    conn.rollback()
                except Exception as e:
                    logging.critical(f"A critical unexpected error occurred: {e}", exc_info=True)
                    conn.rollback()

    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")


if __name__ == "__main__":
    main()