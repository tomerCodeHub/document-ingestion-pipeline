import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

def test_ai():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    try:
        # Use the NEW model name
        model_id = "gemini-embedding-001"
        
        print(f"Connecting to {model_id}...")
        
        result = client.models.embed_content(
            model=model_id,
            contents="Testing the new embedding model.",
            config={
                'output_dimensionality': 768  # Matches your Postgres vector(768)
            }
        )

        vector = result.embeddings[0].values
        print("✅ SUCCESS! The model is active.")
        print(f"✅ Vector length: {len(vector)}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_ai()