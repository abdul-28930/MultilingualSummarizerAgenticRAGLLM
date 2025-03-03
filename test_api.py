from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables
load_dotenv(override=True)

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key starts with: {api_key[:10]}...")

try:
    # Initialize client
    client = OpenAI(api_key=api_key)
    
    # Try a simple completion
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10
    )
    print("Success! Response:", response.choices[0].message.content)
except Exception as e:
    print("Error:", str(e))
