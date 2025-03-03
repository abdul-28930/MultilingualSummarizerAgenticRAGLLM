import os
from dotenv import load_dotenv

# First check without loading .env
print("Before loading .env:")
print(f"OPENAI_API_KEY = {os.environ.get('OPENAI_API_KEY', 'Not set')}")

# Now load .env
load_dotenv(override=True)

# Check after loading
print("\nAfter loading .env:")
print(f"OPENAI_API_KEY = {os.environ.get('OPENAI_API_KEY', 'Not set')}")

# List all environment variables starting with OPENAI
print("\nAll OpenAI-related environment variables:")
for key, value in os.environ.items():
    if 'OPENAI' in key:
        print(f"{key} = {value[:10]}...")
