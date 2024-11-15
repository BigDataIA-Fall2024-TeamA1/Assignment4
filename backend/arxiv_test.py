import requests
import re

# Regex to extract abstract from ArXiv HTML
abstract_pattern = re.compile(
    r'<blockquote class="abstract mathjax">\s*<span class="descriptor">Abstract:</span>\s*(.*?)\s*</blockquote>',
    re.DOTALL
)

def fetch_arxiv_papers(query: str, max_results: int = 5):
    """
    Fetches relevant ArXiv papers based on a query and extracts their abstracts.
    Args:
        query (str): Search query.
        max_results (int): Maximum number of papers to fetch.
    Returns:
        List of tuples (paper title, abstract, link).
    """
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        return f"Error fetching papers: {response.status_code}"
    
    # Parse the response (Atom XML format)
    papers = []
    entries = re.findall(r'<entry>(.*?)</entry>', response.text, re.DOTALL)
    
    for entry in entries:
        title = re.search(r'<title>(.*?)</title>', entry, re.DOTALL).group(1).strip()
        summary = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL).group(1).strip()
        link = re.search(r'<id>(.*?)</id>', entry, re.DOTALL).group(1).strip()
        
        papers.append((title, summary, link))
    
    return papers

# Test ArXiv Fetching
if __name__ == "__main__":
    user_query = input("Enter your query: ").strip()
    # Fetch papers related to the user's query
    results = fetch_arxiv_papers(user_query)

    # Print results
    if isinstance(results, str):
        print("Error:", results)
    else:
        print(f"\nFound {len(results)} papers for query '{user_query}':\n")
        for i, (title, summary, link) in enumerate(results, 1):
            print(f"Paper {i}: {title}\nAbstract: {summary}\nLink: {link}\n{'-'*80}")
