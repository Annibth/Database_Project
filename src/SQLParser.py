import sqlparse
from typing import List

class SQLParser:
    """
    Parses an SQL query to extract table names.
    This is a simplified parser focusing on FROM and JOIN clauses.
    """

    
    def get_tables(self, sql_query: str) -> List[str]:
        """Extracts table names from the FROM and JOIN clauses of an SQL query."""
        tables = set()
        parsed = sqlparse.parse(sql_query)[0]
        from_seen = False
        for token in parsed.tokens:
            if isinstance(token, sqlparse.sql.Where):
                # Stop parsing after the WHERE clause
                break
            if from_seen:
                # This is a simplification. It handles basic "table alias" but not complex subqueries.
                if token.ttype is sqlparse.tokens.Keyword and token.value.upper() not in ['AS', 'ON', 'USING']:
                    continue
                if isinstance(token, sqlparse.sql.Identifier):
                    tables.add(token.get_real_name())
                elif isinstance(token, sqlparse.sql.IdentifierList):
                    for identifier in token.get_identifiers():
                        tables.add(identifier.get_real_name())

            if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'FROM':
                from_seen = True
            # Also capture tables from JOIN clauses
            if token.ttype is sqlparse.tokens.Keyword and 'JOIN' in token.value.upper():
                # The next non-keyword, non-whitespace token should be the table
                next_token_idx = parsed.token_index(token) + 1
                while next_token_idx < len(parsed.tokens):
                    next_token = parsed.tokens[next_token_idx]
                    if next_token.is_whitespace:
                        next_token_idx += 1
                        continue
                    if isinstance(next_token, sqlparse.sql.Identifier):
                        tables.add(next_token.get_real_name())
                    break

        print(f"Parsed tables: {tables} from query.")
        return sorted(list(tables)) # Return a sorted list for consistency
