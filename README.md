# Database_Project

### Install PostgreSQL

On MacOS do the following steps:

1) ```brew install postgresql```
2) ```brew services start postgresql```

Verify the working installation by: ```psql postgres```

### Download and unpack JOB-Data

1) **Create new Directory**
    ```shell
    mkdir jobdata && cd jobdata
    ```

2) **download the data**
    ```shell
    curl -OL https://bonsai.cedardb.com/job/imdb.tgz
    ```

3) **unpack it**
    ```shell
    tar -zxvf imdb.tgz
    ```
> It includes all data and the database scheme ("schematext.sql")

 

### Create and fill JOB Movie-Database

1) **start postgres**
    ```shell
    psql postgres
    ```
2) **Create database**
    ```bash
    CREATE DATABASE imdb;
    ```
3) **Connect to database**

    ```bash
    \c imdb
    ```

    >**Note:** If this is not working, you may have to enable connections for this database, try: 
    >```shell
    >SELECT datname, datallowconn FROM pg_database;
    >``` 
    >If this shows "f" for your database, try:
    >```shell
    >UPDATE pg_database SET datallowconn = true WHERE datname = 'imdb';
    >```

4) **Import scheme**
    ```shell
    \i <your_path/jobdata/schematext.sql>
    ```

5) **Load with data:** 

    You find a script named "loader.py" at "Database_Project/database"

    - Change the path (line 4) and username (line 11), then execute
    