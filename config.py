import os
from dotenv import load_dotenv

# Load environment variables from system or .env file if available
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY environment variable.")
if not TAVILY_API_KEY:
    raise ValueError("Missing TAVILY_API_KEY environment variable.")