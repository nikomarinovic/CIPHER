import base64
import math
import requests

from bs4 import BeautifulSoup
from urllib.parse import unquote, urlparse, parse_qs

from brain.nlp_utils import tokenize, phrase_proximity_bonus


class InternetEngine:
    """
    Pure retrieval engine: search -> rank candidates -> fetch best pages ->
    extract text -> rank again with full content -> return structured
    results. It does NOT decide what language to answer in and does NOT
    generate an answer — that is CIPHERBrain / the synthesizer's job.
    """

    DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/"
    BING_URL = "https://www.bing.com/search"

    DEBUG = False

    CANDIDATE_POOL = 20   
    FETCH_TOP_N = 7         
    MIN_CONTENT_LENGTH = 100  

    RESULT_ABS_FLOOR = 0.12  
    RESULT_REL_FLOOR = 0.40  

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/139.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    def search(self, query: str, limit: int = 5) -> list[dict]:
        self._debug(f"Search query: {query}")

        candidates = self._search_request(query, self.CANDIDATE_POOL)
        candidates = self._dedupe(candidates)

        self._debug(f"Search results: {len(candidates)}")

        if not candidates:
            return []

        query_tokens = tokenize(query)

        metadata_texts = [
            f"{candidate.get('title', '')} {candidate.get('snippet', '')} {candidate.get('url', '')}"
            for candidate in candidates
        ]
        metadata_df = self._document_frequency(query_tokens, metadata_texts)

        for candidate in candidates:
            candidate["prelim_score"] = self._score_candidate_metadata(
                query_tokens, candidate, metadata_df, len(metadata_texts)
            )

        candidates.sort(key=lambda item: item["prelim_score"], reverse=True)

        fetch_pool = candidates[:min(len(candidates), max(limit + 2, self.FETCH_TOP_N))]

        enriched = []

        for candidate in fetch_pool:
            self._debug(f"Fetching: {candidate['title']}")

            content = self._read_page(candidate["url"])

            if not content or len(content) < self.MIN_CONTENT_LENGTH:
                self._debug(f"Page rejected: {candidate['title']}")
                continue

            candidate["content"] = content
            enriched.append(candidate)
            self._debug(f"Page accepted: {candidate['title']}")

        if not enriched:
            return []

        content_texts = [item["content"] for item in enriched]
        content_df = self._document_frequency(query_tokens, content_texts)

        for candidate in enriched:
            content_score = self._score_text(
                query_tokens, candidate["content"], content_df, len(content_texts), weighted=True
            )
            content_score *= phrase_proximity_bonus(query_tokens, candidate["content"])

            candidate["relevance"] = (candidate["prelim_score"] * 0.3) + (content_score * 0.7)
            del candidate["prelim_score"]

        enriched.sort(key=lambda item: item["relevance"], reverse=True)

        top_score = enriched[0]["relevance"]
        floor = max(self.RESULT_ABS_FLOOR, top_score * self.RESULT_REL_FLOOR) if top_score > 0 else 0

        filtered = [item for item in enriched if item["relevance"] >= floor][:limit]

        self._debug(f"Final sources: {len(filtered)}")

        return filtered


    def _score_candidate_metadata(
        self,
        query_tokens: list[str],
        candidate: dict,
        document_frequency: dict,
        total_documents: int,
    ) -> float:
        """Score a search result using only title/snippet/URL — before we've
        fetched the actual page. Generic, IDF-weighted token overlap: a
        query word that shows up in nearly every candidate (e.g. an
        incidental match like "tko" hitting a "TKO" brand name across half
        the results) carries little weight, while a word that only a
        couple of candidates contain (e.g. "nikola") carries a lot. This
        stays entirely statistical — nothing entity- or language-specific."""
        if not query_tokens:
            return 0.0

        title_tokens = set(tokenize(candidate.get("title", "")))
        snippet_tokens = set(tokenize(candidate.get("snippet", "")))
        url_tokens = set(self._tokenize_url(candidate.get("url", "")))

        def weighted_overlap(token_set: set) -> float:
            total_idf = 0.0
            matched_idf = 0.0

            for token in query_tokens:
                idf = math.log(
                    (total_documents + 1) / (document_frequency.get(token, 0) + 1)
                ) + 1
                total_idf += idf

                if token in token_set:
                    matched_idf += idf

            if total_idf == 0:
                return 0.0

            return matched_idf / total_idf

        title_overlap = weighted_overlap(title_tokens)
        snippet_overlap = weighted_overlap(snippet_tokens)
        url_overlap = weighted_overlap(url_tokens)

        score = (title_overlap * 0.60) + (snippet_overlap * 0.25) + (url_overlap * 0.15)

        if title_overlap >= 0.95:
            score *= 1.25

        combined_text = f"{candidate.get('title', '')} {candidate.get('snippet', '')}"
        score *= phrase_proximity_bonus(query_tokens, combined_text)

        return score

    def _tokenize_url(self, url: str) -> list[str]:
        try:
            parsed = urlparse(url)
            text = f"{parsed.netloc} {parsed.path}"
            text = text.replace("-", " ").replace("_", " ").replace("/", " ").replace(".", " ")
            return tokenize(text)
        except Exception:
            return []

    def _document_frequency(self, tokens: list[str], texts: list[str]) -> dict:
        document_frequency = {}

        for token in set(tokens):
            count = 0

            for text in texts:
                if token in text.lower():
                    count += 1

            document_frequency[token] = count

        return document_frequency

    def _score_text(
        self,
        query_tokens: list[str],
        text: str,
        document_frequency: dict,
        total_documents: int,
        weighted: bool = False
    ) -> float:
        if not query_tokens or not text:
            return 0.0

        text_lower = text.lower()
        score = 0.0
        max_possible = 0.0

        for token in query_tokens:
            idf = math.log(
                (total_documents + 1) / (document_frequency.get(token, 0) + 1)
            ) + 1

            max_possible += idf
            count = text_lower.count(token)

            if count:
                contribution = idf

                if weighted:
                    contribution *= 1 + min(count, 5) * 0.05

                score += contribution

        if max_possible == 0:
            return 0.0

        return score / max_possible

    def _dedupe(self, candidates: list[dict]) -> list[dict]:
        seen = set()
        deduped = []

        for candidate in candidates:
            key = candidate.get("url", "").rstrip("/")

            if not key or key in seen:
                continue

            seen.add(key)
            deduped.append(candidate)

        return deduped

    BLOCK_MARKERS = [
        "unusual traffic",
        "verify you are human",
        "enable javascript and cookies",
        "detected unusual activity",
        "are you a robot",
        "recaptcha",
    ]

    def _looks_blocked(self, html_text: str) -> bool:
        lowered = html_text.lower()
        return any(marker in lowered for marker in self.BLOCK_MARKERS)

    def _search_request(self, query: str, limit: int) -> list[dict]:
        results = self._search_bing(query, limit)

        if results:
            return results

        self._debug("Bing returned nothing, falling back to DuckDuckGo (html)")
        results = self._search_duckduckgo(query, limit)

        if results:
            return results

        self._debug("DuckDuckGo (html) returned nothing, falling back to DuckDuckGo (lite)")
        return self._search_duckduckgo_lite(query, limit)

    def _search_bing(self, query: str, limit: int) -> list[dict]:
        try:
            response = requests.get(
                self.BING_URL,
                params={
                    "q": query,
                    "count": limit,
                    "mkt": "en-US",
                    "setlang": "en",
                },
                headers=self.HEADERS,
                timeout=15
            )

            self._debug(f"Bing status: {response.status_code}, length: {len(response.text)}")

            response.raise_for_status()

        except requests.RequestException as error:
            self._debug(f"Bing request failed: {error}")
            return []

        if self._looks_blocked(response.text):
            self._debug("Bing appears to be showing a bot-check/consent page instead of results")
            self._dump_debug_html("bing", response.text)
            return []

        soup = BeautifulSoup(response.text, "html.parser")

        page_title = soup.title.get_text(strip=True) if soup.title else "(no <title>)"
        self._debug(f"Bing page title: {page_title}")

        results = []

        result_blocks = soup.select("li.b_algo")
        self._debug(f"Bing: li.b_algo matched {len(result_blocks)}")

        if not result_blocks:
            container = soup.select_one("#b_results")

            if container:
                result_blocks = container.select("li")
                self._debug(f"Bing: #b_results li matched {len(result_blocks)}")

        for result in result_blocks:
            link = result.select_one("h2 a")

            if not link:
                continue

            title = link.get_text(" ", strip=True)
            raw_url = link.get("href")

            if not title or not raw_url:
                continue

            url = self._clean_bing_url(raw_url)

            if not url:
                continue

            snippet_element = result.select_one(".b_caption p") or result.select_one("p")
            snippet = snippet_element.get_text(" ", strip=True) if snippet_element else ""

            results.append({
                "title": title,
                "url": url,
                "snippet": snippet
            })

            if len(results) >= limit:
                break

        if not results:
            self._debug("Bing: falling back to generic h2>a scan")

            for heading in soup.select("h2"):
                link = heading.find("a", href=True)

                if not link:
                    continue

                title = link.get_text(" ", strip=True)
                raw_url = link.get("href")

                if not title or not raw_url:
                    continue

                url = self._clean_bing_url(raw_url)

                if not url or "bing.com" in url or "microsoft.com" in url:
                    continue

                snippet = ""
                next_p = heading.find_next("p")

                if next_p:
                    snippet = next_p.get_text(" ", strip=True)

                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet
                })

                if len(results) >= limit:
                    break

            self._debug(f"Bing: generic scan found {len(results)}")

        if not results:
            self._dump_debug_html("bing", response.text)

        return results

    def _clean_bing_url(self, url: str) -> str:
        if not url:
            return ""

        if "bing.com/ck/a" in url:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            encoded_url = params.get("u")

            if encoded_url:
                encoded = encoded_url[0]

                if encoded.startswith("a1"):
                    encoded = encoded[2:]

                decoded = self._decode_bing_base64(encoded)

                if decoded:
                    return decoded

                try:
                    decoded = unquote(encoded)

                    if decoded.startswith(("http://", "https://")):
                        return decoded

                except Exception:
                    pass

            for key in ("r", "url"):
                if key in params:
                    try:
                        candidate = unquote(params[key][0])

                        if candidate.startswith(("http://", "https://")):
                            return candidate

                    except Exception:
                        pass

            return ""

        if url.startswith("//"):
            return f"https:{url}"

        if url.startswith(("http://", "https://")):
            return url

        return ""

    def _decode_bing_base64(self, encoded: str) -> str:
        try:
            padded = encoded + "=" * (-len(encoded) % 4)
            decoded_bytes = base64.urlsafe_b64decode(padded)
            decoded = decoded_bytes.decode("utf-8", errors="ignore")

            if decoded.startswith(("http://", "https://")):
                return decoded

        except Exception:
            pass

        return ""

    def _search_duckduckgo(self, query: str, limit: int) -> list[dict]:
        try:
            response = requests.post(
                self.DUCKDUCKGO_URL,
                data={"q": query, "kl": "us-en"},
                headers=self.HEADERS,
                timeout=15
            )

            self._debug(f"DuckDuckGo (html) status: {response.status_code}, length: {len(response.text)}")

            response.raise_for_status()

        except requests.RequestException as error:
            self._debug(f"DuckDuckGo request failed: {error}")
            return []

        if self._looks_blocked(response.text):
            self._debug("DuckDuckGo appears to be showing a bot-check page instead of results")
            self._dump_debug_html("ddg_html", response.text)
            return []

        soup = BeautifulSoup(response.text, "html.parser")

        results = []

        for result in soup.select("div.result"):
            link = result.select_one("a.result__a")

            if not link:
                continue

            title = link.get_text(" ", strip=True)
            raw_url = link.get("href")

            if not title or not raw_url:
                continue

            url = self._clean_duckduckgo_url(raw_url)

            if not url:
                continue

            snippet_element = result.select_one("a.result__snippet") or result.select_one(".result__snippet")
            snippet = snippet_element.get_text(" ", strip=True) if snippet_element else ""

            results.append({
                "title": title,
                "url": url,
                "snippet": snippet
            })

            if len(results) >= limit:
                break

        if not results:
            self._dump_debug_html("ddg_html", response.text)

        return results

    def _clean_duckduckgo_url(self, url: str) -> str:
        if not url:
            return ""

        if url.startswith("//"):
            url = f"https:{url}"

        if "duckduckgo.com/l/" in url:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            encoded_url = params.get("uddg")

            if encoded_url:
                try:
                    decoded = unquote(encoded_url[0])

                    if decoded.startswith(("http://", "https://")):
                        return decoded

                except Exception:
                    pass

            return ""

        if url.startswith(("http://", "https://")):
            return url

        return ""

    def _search_duckduckgo_lite(self, query: str, limit: int) -> list[dict]:
        """DuckDuckGo's plain-HTML 'lite' endpoint. It has much lighter
        bot-detection than the main html.duckduckgo.com endpoint, so it's
        a useful last-resort fallback."""
        try:
            response = requests.get(
                "https://lite.duckduckgo.com/lite/",
                params={"q": query, "kl": "us-en"},
                headers=self.HEADERS,
                timeout=15
            )

            self._debug(f"DuckDuckGo (lite) status: {response.status_code}, length: {len(response.text)}")

            response.raise_for_status()

        except requests.RequestException as error:
            self._debug(f"DuckDuckGo (lite) request failed: {error}")
            return []

        if self._looks_blocked(response.text):
            self._debug("DuckDuckGo Lite appears to be showing a bot-check page instead of results")
            return []

        soup = BeautifulSoup(response.text, "html.parser")

        results = []

        for link in soup.select("a.result-link"):
            title = link.get_text(" ", strip=True)
            raw_url = link.get("href")

            if not title or not raw_url:
                continue

            url = raw_url if raw_url.startswith(("http://", "https://")) else ""

            if not url:
                continue

            snippet = ""
            row = link.find_parent("tr")

            if row:
                snippet_row = row.find_next_sibling("tr")

                if snippet_row:
                    snippet_cell = snippet_row.select_one(".result-snippet")

                    if snippet_cell:
                        snippet = snippet_cell.get_text(" ", strip=True)

            results.append({
                "title": title,
                "url": url,
                "snippet": snippet
            })

            if len(results) >= limit:
                break

        return results


    def _read_page(self, url: str, max_chars: int = 8000) -> str:
        try:
            response = requests.get(
                url,
                headers=self.HEADERS,
                timeout=10,
                allow_redirects=True
            )

            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()

            if (
                "text/html" not in content_type
                and "application/xhtml+xml" not in content_type
            ):
                return ""

            soup = BeautifulSoup(response.text, "html.parser")
            content = self._extract_main_text(soup)
            content = " ".join(content.split())

            return content[:max_chars]

        except (requests.RequestException, UnicodeDecodeError) as error:
            self._debug(f"Page fetch failed ({url}): {error}")
            return ""

    def _extract_main_text(self, soup: BeautifulSoup) -> str:
        for element in soup([
            "script", "style", "nav", "header", "footer",
            "aside", "form", "noscript", "svg", "iframe", "canvas"
        ]):
            element.decompose()

        for element in soup.select(
            '[role="navigation"], [role="banner"], [role="complementary"], [role="contentinfo"], '
            '[class*="cookie" i], [id*="cookie" i], [class*="advert" i], [id*="advert" i]'
        ):
            element.decompose()

        main_container = soup.find("article") or soup.find("main")

        if main_container:
            paragraphs = main_container.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        paragraph_text = " ".join(
            paragraph.get_text(" ", strip=True)
            for paragraph in paragraphs
            if paragraph.get_text(strip=True)
        )

        if len(paragraph_text) >= 200:
            return paragraph_text

        fallback_container = main_container or soup
        return fallback_container.get_text(" ", strip=True)


    def _dump_debug_html(self, name: str, text: str) -> None:
        if not self.DEBUG:
            return

        try:
            path = f"/tmp/cipher_debug_{name}.html"

            with open(path, "w", encoding="utf-8") as file:
                file.write(text)

            self._debug(f"Saved raw response to {path} for inspection")

        except Exception as error:
            self._debug(f"Could not save debug HTML: {error}")

    def _debug(self, message: str) -> None:
        if self.DEBUG:
            print(f"[DEBUG] {message}")


if __name__ == "__main__":
    import sys
    import os

    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from brain.internet import InternetEngine 

    query = " ".join(sys.argv[1:]) or "nikola tesla"
    engine = InternetEngine()

    results = engine.search(query, limit=5)
    print(f"\nfinal filtered results: {len(results)}")

    for result in results:
        print(result["title"])
        print(result["url"])
        print(f"relevance: {result['relevance']:.3f}")
        print(result.get("content", "")[:200])
        print("---")