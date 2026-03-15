import sqlite3
import pandas as pd

def check_db():
    conn = sqlite3.connect('data/raw/BTCUSDT.db')
    
    # Check tables
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql_query(query, conn)
    print("Tables found:")
    print(tables)
    print("\n")
    
    if len(tables) > 0:
        table_name = tables.iloc[0]['name']
        print(f"Sample data from {table_name}:")
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
        print(df.head())
        print("\nColumns:")
        print(df.columns.tolist())
        
    conn.close()

if __name__ == "__main__":
    check_db()
