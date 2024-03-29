import pandas as pd
import sqlite3

db_file = "mlb.db"
conn = sqlite3.connect(db_file)

df = pd.read_csv("data/mlb_elo.csv")
df.to_sql("elo", conn, if_exists="replace", index=False)

df = pd.read_csv("data/baseballdatabank-2023.1/core/Teams.csv")
df.to_sql("teams", conn, if_exists="replace", index=False)

conn.close()
