import os
import psycopg2
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

def test_connection():
    try:
        # Get the URL from .env
        url = os.getenv("POSTGRES_URL")
        
        # Try to connect
        conn = psycopg2.connect(url)
        print("✅ Connection Successful!")
        
        # Check if the table exists
        cur = conn.cursor()
        cur.execute("SELECT * FROM document_vectors LIMIT 1;")
        print("✅ Table 'document_vectors' is accessible!")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Connection Failed: {e}")

if __name__ == "__main__":
    test_connection()