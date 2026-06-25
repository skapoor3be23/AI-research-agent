import requests

def web_search(query):
    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    results = []
    
    # Main abstract/summary if available
    if data.get("AbstractText"):
        results.append(data["AbstractText"])
    
    # Related topics (usually short snippets)
    for topic in data.get("RelatedTopics", [])[:3]:
        if isinstance(topic, dict) and "Text" in topic:
            results.append(topic["Text"])
    
    if not results:
        return "No web results found for this query."
    
    return "\n".join(results)

# Test it
if __name__ == "__main__":
    query = input("Enter a search query: ")
    result = web_search(query)
    print("\n--- WEB SEARCH RESULT ---")
    print(result)