# Database_Project

### Install PostgreSQL

On MacOS do the following steps:

1) ```brew install postgresql```
2) ```brew services start postgresql```

Verify the working installation by: ```psql postgres```

### Install TPC-H dbgen tool for generating data tables

Clone the repository and clone it by typinh: 
```shell
git clone https://github.com/electrum/tpch-dbgen
cd tpch-dbgen

# build the dbgen binary
make

# generate data for tables 
./dbgen -s 1 # scale factor 1 is ~1 GB

# create database 
psql postgres 
CREATE DATABASE tpch;
\c tpch

# load in the scheme for database 
psql tpch < data/tpch-schema.sql
```
To load the generated data inside the database tables you can use the predefinesd loading script. 

Make sure to assign executable rights to this script:
```shell
sudo chmod +x data/load_tpch_data.sh
```
Now run the script ad verify is data was transferred correctly: 
```shell 
# clean and load data tables
./data/load_tpch_data.sh

# verify successfull loading
psql -U _username_ -d tcph
SELECT COUNT(*) FROM lineitem;
# you should see around 6 million rows