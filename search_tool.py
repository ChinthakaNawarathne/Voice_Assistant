from tavily import TavilyClient
import config

# Words that indicate a question doesn't need a live web search
CONVERSATIONAL_KEYWORDS = {"hello", "hi", "hey", "how are you", "who are you", "thank you", "thanks", "bye", "goodbye"}

def search_web(query: str) -> str:
    """Executes a live search via Tavily API only if the query requires real-time data."""
    clean_query = query.lower().strip(".,!?")
    
    # Fast exit for simple casual conversation
    if clean_query in CONVERSATIONAL_KEYWORDS or len(clean_query.split()) <= 2:
        return "No search needed. This is a simple greeting or casual conversational remark."

    try:
        if not config.TAVILY_API_KEY:
            return "No search tool available."

        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        # Use 'basic' search and limit results to 2 to get data back faster
        response = client.search(query=query, search_depth="basic", max_results=2)
        
        results = [item.get("content", "") for item in response.get("results", [])]
        return "\n".join(results) if results else "No specific live web data found."
    except Exception:
        return "Search tool temporarily unavailable."