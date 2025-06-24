-- Query 1: 3-way join
SELECT *
FROM customer
JOIN orders ON customer.c_custkey = orders.o_custkey
JOIN lineitem ON orders.o_orderkey = lineitem.l_orderkey;

-- Query 2: 4-way join
SELECT *
FROM region
JOIN nation ON region.r_regionkey = nation.n_regionkey
JOIN customer ON nation.n_nationkey = customer.c_nationkey
JOIN orders ON customer.c_custkey = orders.o_custkey;

-- Query 3: 5-way join
SELECT *
FROM part
JOIN partsupp ON part.p_partkey = partsupp.ps_partkey
JOIN supplier ON partsupp.ps_suppkey = supplier.s_suppkey
JOIN lineitem ON part.p_partkey = lineitem.l_partkey AND supplier.s_suppkey = lineitem.l_suppkey
JOIN orders ON lineitem.l_orderkey = orders.o_orderkey;

-- Query 4: 6-way join
SELECT *
FROM customer
JOIN orders ON customer.c_custkey = orders.o_custkey
JOIN lineitem ON orders.o_orderkey = lineitem.l_orderkey
JOIN partsupp ON lineitem.l_suppkey = partsupp.ps_suppkey AND lineitem.l_partkey = partsupp.ps_partkey
JOIN supplier ON partsupp.ps_suppkey = supplier.s_suppkey
JOIN nation ON supplier.s_nationkey = nation.n_nationkey;

-- Query 5: 6-way join
SELECT *
FROM part
JOIN partsupp ON part.p_partkey = partsupp.ps_partkey
JOIN supplier ON partsupp.ps_suppkey = supplier.s_suppkey
JOIN lineitem ON part.p_partkey = lineitem.l_partkey AND supplier.s_suppkey = lineitem.l_suppkey
JOIN orders ON lineitem.l_orderkey = orders.o_orderkey
JOIN customer ON orders.o_custkey = customer.c_custkey;

-- Query 6: 6-way join
SELECT *
FROM region
JOIN nation ON region.r_regionkey = nation.n_regionkey
JOIN supplier ON nation.n_nationkey = supplier.s_nationkey
JOIN partsupp ON supplier.s_suppkey = partsupp.ps_suppkey
JOIN part ON partsupp.ps_partkey = part.p_partkey
JOIN lineitem ON part.p_partkey = lineitem.l_partkey AND partsupp.ps_suppkey = lineitem.l_suppkey;