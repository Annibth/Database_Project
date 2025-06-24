
from typing import List, Tuple, Dict, Any, Set
import time

import psycopg2

class PostgresExecutor:
    """Handles executing queries against the PostgreSQL database and measuring latency."""
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self._connection = None

    def _get_connection(self):
        """Establishes or reuses a database connection."""
        if self._connection is None or self._connection.closed:
            try:
                self._connection = psycopg2.connect(**self.config)
            except psycopg2.OperationalError as e:
                print(f"Error connecting to PostgreSQL: {e}")
                raise
        return self._connection

    def execute_query(self, query: str) -> float:
        """
        Executes a given SQL query and returns the execution time in seconds.
        Returns float('inf') on error.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Use EXPLAIN ANALYZE to get the actual execution time from the planner
        explain_query = f"EXPLAIN ANALYZE {query}"
        
        try:
            start_time = time.perf_counter()
            cursor.execute(explain_query)
            result = cursor.fetchall()
            end_time = time.perf_counter()

            # The last line of EXPLAIN ANALYZE contains the execution time
            execution_time_line = [line[0] for line in result if "Execution Time" in line[0]]
            if execution_time_line:
                # Example line: "Execution Time: 123.456 ms"
                time_str = execution_time_line[0].split(":")[1].strip()
                value, unit = time_str.split()
                if unit == 'ms':
                    return float(value) / 1000.0
                else: # Assuming seconds if not ms
                    return float(value)
            else:
                # Fallback to wall-clock time if parsing fails
                print("Warning: Could not parse execution time from EXPLAIN ANALYZE. Using wall-clock time.")
                return end_time - start_time

        except psycopg2.Error as e:
            print(f"Error executing query: {e}")
            conn.rollback()
            return float('inf') # Return a very high cost for failed queries
        finally:
            cursor.close()

    def close(self):
        """Closes the database connection if it's open."""
        if self._connection and not self._connection.closed:
            self._connection.close()