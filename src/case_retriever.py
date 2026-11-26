"""Automated Indian infrastructure arbitration case retriever (Playwright + Google Search API)."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import requests


GOOGLE_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"

DEFAULT_SEARCH_QUERIES = [
    'site:indiankanoon.org "Arbitration and Conciliation Act" "NHAI" "Crore" "award"',
    'site:indiankanoon.org "Section 34" "DMRC" OR "RVNL" "arbitral award" "Crore"',
    'site:indiankanoon.org "Section 34" "NTPC" OR "Power Grid" "arbitral tribunal" "Crore"',
    'site:indiankanoon.org "Section 34" "Port Trust" OR "Airports Authority" "arbitral award"',
    'site:indiankanoon.org "Section 34" "Jal Board" OR "Municipal Corporation" "arbitration" "EPC"',
]


class ArbitrationCaseRetriever:
    """
    Automates discovery and scraping of Indian infrastructure arbitration judgments
    using the Google Custom Search JSON API + headless Playwright page extraction.
    """

    def __init__(
        self,
        google_api_key: Optional[str] = None,
        google_cse_id: Optional[str] = None,
        timeout_ms: int = 20000,
    ) -> None:
        self.api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.cse_id = google_cse_id or os.getenv("GOOGLE_CSE_ID")
        self.timeout_ms = timeout_ms

    def search_case_urls(self, queries: Optional[List[str]] = None, results_per_query: int = 10) -> List[Dict[str, str]]:
        """Query Google Custom Search API for candidate Indian infrastructure arbitration URLs."""
        if not self.api_key or not self.cse_id:
            raise EnvironmentError(
                "Live retrieval requires GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables."
            )

        discovered: List[Dict[str, str]] = []
        seen = set()
        for q in queries or DEFAULT_SEARCH_QUERIES:
            resp = requests.get(
                GOOGLE_CSE_ENDPOINT,
                params={"key": self.api_key, "cx": self.cse_id, "q": q, "num": min(results_per_query, 10)},
                timeout=15,
            )
            resp.raise_for_status()
            for item in resp.json().get("items", []):
                url = item.get("link", "")
                if url and url not in seen:
                    seen.add(url)
                    discovered.append({
                        "title": item.get("title", ""),
                        "url": url,
                        "snippet": item.get("snippet", ""),
                    })
        return discovered

    def scrape_urls_with_playwright(self, urls: List[str]) -> List[Dict[str, str]]:
        """Render and extract clean judgment text using headless Playwright."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise ImportError(
                "Playwright is required for live browser scraping. Install via: "
                "pip install playwright && playwright install chromium"
            ) from exc

        documents: List[Dict[str, str]] = []
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (compatible; IIMA-LegalResearch/1.0)"
            )
            page = context.new_page()
            for url in urls:
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                    text = page.locator("div.judgments, article, main, body").first.inner_text()
                    documents.append({"url": url, "text": text.strip()})
                except Exception:
                    continue
            browser.close()
        return documents
