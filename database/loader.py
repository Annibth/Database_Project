import psycopg2
import os

path = "/Users/anna-mariabothin/Documents/Studium/Sommersemester_25/DB_Projekt/jobdata"
dir = [x.name for x in os.scandir(path) if "csv" in x.name]


# Verbindung zur DB
conn = psycopg2.connect(
    dbname="imdb",
    user="anna-mariabothin",
    host="localhost",
    port="5432"
)

cur = conn.cursor()

for file in dir:
    table = file[:-4]
    cur.execute(f"SELECT EXISTS (SELECT 1 FROM {table} LIMIT 1);")
    if cur.fetchone()[0]:
        print(f"Tabelle {table} ist bereits befüllt")
    else:
        cur.execute(fr"copy {table} from '{path}/{file}' delimiter ',' CSV NULL '' ESCAPE '\' HEADER;")
        print(f"Tabelle {table} erfolgreich befüllt!")

conn.commit()
cur.close()
conn.close()