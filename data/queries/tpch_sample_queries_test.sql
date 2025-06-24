-- Query 1: 3-way join
SELECT *
FROM customer
JOIN orders ON customer.c_custkey = orders.o_custkey
JOIN lineitem ON orders.o_orderkey = lineitem.l_orderkey;