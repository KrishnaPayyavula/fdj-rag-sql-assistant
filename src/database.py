import sqlite3
import pandas as pd
import os
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path: str = "db/products.db"):
        self.db_path = db_path
        self.ensure_db_directory()
        
    def ensure_db_directory(self):
        """Create database directory if it doesn't exist."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
    
    def create_schema(self):
        """Create the products table schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Drop table if exists for fresh start
        cursor.execute("DROP TABLE IF EXISTS products")
        
        # Create products table
        cursor.execute("""
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                turnover REAL,
                launch_date DATE,
                country TEXT,
                segment TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        print("Schema created successfully!")
    
    def load_data(self, csv_path: str = "data/csv/products.csv"):
        """Load data from CSV into the database."""
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Convert launch_date to proper date format
        df['launch_date'] = pd.to_datetime(df['launch_date'], format='%d/%m/%Y')
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        
        # Load data into products table
        df.to_sql('products', conn, if_exists='replace', index=False)
        
        conn.close()
        print(f"Loaded {len(df)} records into the database!")
    
    def test_connection(self):
        """Test database connection and show sample data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get table info
        cursor.execute("PRAGMA table_info(products)")
        columns = cursor.fetchall()
        print("\nTable Schema:")
        for col in columns:
            print(f"  {col[1]} ({col[2]})")
        
        # Get sample data
        cursor.execute("SELECT * FROM products LIMIT 5")
        rows = cursor.fetchall()
        print("\nSample Data:")
        for row in rows:
            print(f"  {row}")
        
        # Get statistics
        cursor.execute("SELECT COUNT(*) FROM products")
        count = cursor.fetchone()[0]
        print(f"\nTotal records: {count}")
        
        conn.close()

if __name__ == "__main__":
    # Initialize database
    db_manager = DatabaseManager()
    db_manager.create_schema()
    db_manager.load_data()
    db_manager.test_connection()