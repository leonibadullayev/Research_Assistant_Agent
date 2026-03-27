import arxiv
import requests
import fitz  # PyMuPDF
import os
from typing import List, Dict
from utils import cache


def search_arxiv(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search arXiv for papers matching the query.
    Returns a list of dicts with keys: title, authors, abstract, pdf_url, published.
    """
    cache_key = f"arxiv_{query}_{max_results}"
    if cache_key in cache:
        return cache[cache_key]

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = []
    for paper in client.results(search):
        results.append({
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "abstract": paper.summary,
            "pdf_url": paper.pdf_url,
            "published": paper.published.isoformat(),
        })
    cache[cache_key] = results
    return results


def fetch_pdf_text(pdf_url: str) -> str:
    """Download a PDF from a URL and extract its text."""
    try:
        response = requests.get(pdf_url, timeout=10)
        with open("temp.pdf", "wb") as f:
            f.write(response.content)
        doc = fitz.open("temp.pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text[:10000]
    except Exception as e:
        return f"Error fetching PDF: {e}"


def add_to_zotero(paper_title: str, authors: List[str], year: int, url: str) -> str:
    """Add a paper to your Zotero library."""
    api_key = os.getenv("ZOTERO_API_KEY")
    library_id = os.getenv("ZOTERO_LIBRARY_ID")
    if not api_key or not library_id:
        return "Zotero credentials not set."

    headers = {
        "Zotero-API-Key": api_key,
        "Content-Type": "application/json",
    }
    data = {
        "items": [{
            "itemType": "journalArticle",
            "title": paper_title,
            "creators": [{"creatorType": "author", "name": a} for a in authors],
            "date": str(year),
            "url": url,
        }]
    }
    response = requests.post(
        f"https://api.zotero.org/users/{library_id}/items",
        headers=headers,
        json=data,
    )
    if response.status_code in (200, 201):
        return f"Added {paper_title} to Zotero."
    return f"Failed to add: {response.text}"