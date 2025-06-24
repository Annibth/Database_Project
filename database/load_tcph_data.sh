#!/bin/bash

# ---- CONFIG ----
DB_NAME="tpch"
DB_USER="eliaruhle"   
DATA_DIR="/Users/eliaruhle/Documents/tpch-dbgen"  

# List of tables in correct order (to satisfy foreign key dependencies if any)
TABLES=("region" "nation" "supplier" "customer" "part" "partsupp" "orders" "lineitem")

# ---- CLEAN .tbl FILES ----
echo "Cleaning .tbl files to remove trailing delimiters..."

mkdir -p "$DATA_DIR/clean"

for tbl in "${TABLES[@]}"; do
    sed 's/|$//' "$DATA_DIR/$tbl.tbl" > "$DATA_DIR/clean/${tbl}.csv"
done

# ---- LOAD INTO POSTGRES ----
echo "Loading data into PostgreSQL database: $DB_NAME"

for tbl in "${TABLES[@]}"; do
    echo "Loading $tbl..."
    psql -U "$DB_USER" -d "$DB_NAME" -c "\COPY $tbl FROM '$DATA_DIR/clean/${tbl}.csv' WITH (FORMAT csv, DELIMITER '|', NULL '')"
done

echo "All data loaded successfully!"
