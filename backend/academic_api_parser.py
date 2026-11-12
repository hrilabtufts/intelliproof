"""Async search clients for arXiv, Semantic Scholar, and PubMed.

The service normalizes provider records into :class:`NormalizedPaper`. It
reuses one HTTP connection pool, supports injected ``httpx`` clients, and
returns partial results from combined searches when one provider is unavailable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import threading
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote, urlparse
from weakref import WeakKeyDictionary

import httpx

if __package__:
    from .schemas.academic_api_schema import (
        AcademicSearchResponse,
        AcademicSource,
        NormalizedPaper,
    )
else:  # Support running/importing backend modules directly from backend/.
    from schemas.academic_api_schema import (
        AcademicSearchResponse,
        AcademicSource,
        NormalizedPaper,
    )


ARXIV_API_URL = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
MAX_RESULTS_PER_PROVIDER = 100
DEFAULT_TIMEOUT_SECONDS = 15.0
# Conservative spacing follows each provider's published rate guidance.
ARXIV_MIN_REQUEST_INTERVAL_SECONDS = 3.05
SEMANTIC_SCHOLAR_MIN_REQUEST_INTERVAL_SECONDS = 1.05
PUBMED_MIN_REQUEST_INTERVAL_NO_KEY_SECONDS = 0.34
PUBMED_MIN_REQUEST_INTERVAL_WITH_KEY_SECONDS = 0.11
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"
_QUERY_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "for",
    "from", "in", "into", "is", "of", "on", "or", "the", "to", "versus",
    "vs", "was", "were", "with", "without",
}


@dataclass
class _ProviderRateLimiter:
    lock: asyncio.Lock
    last_request_at: float | None = None
    blocked_until: float = 0.0


# Shared across client instances on the same backend event loop.
_PROVIDER_RATE_LIMITERS: WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[tuple[str, str], _ProviderRateLimiter]
] = WeakKeyDictionary()
_PROVIDER_RATE_LIMITERS_GUARD = threading.Lock()


def _get_provider_rate_limiter(
    source: AcademicSource,
    credential: str | None = None,
) -> _ProviderRateLimiter:
    loop = asyncio.get_running_loop()
    credential_scope = (
        hashlib.sha256(credential.encode("utf-8")).hexdigest()
        if credential
        else "public"
    )
    key = (source, credential_scope)
    with _PROVIDER_RATE_LIMITERS_GUARD:
        loop_limiters = _PROVIDER_RATE_LIMITERS.get(loop)
        if loop_limiters is None:
            loop_limiters = {}
            _PROVIDER_RATE_LIMITERS[loop] = loop_limiters
        limiter = loop_limiters.get(key)
        if limiter is None:
            limiter = _ProviderRateLimiter(lock=asyncio.Lock())
            loop_limiters[key] = limiter
        return limiter


async def _wait_for_provider_request_slot(
    limiter: _ProviderRateLimiter,
    min_interval: float,
) -> None:
    """Wait for a provider slot while holding its lock through the request."""
    loop = asyncio.get_running_loop()
    now = loop.time()
    next_slot_at = limiter.blocked_until
    if limiter.last_request_at is not None:
        next_slot_at = max(next_slot_at, limiter.last_request_at + min_interval)
    if next_slot_at > now:
        await asyncio.sleep(next_slot_at - now)
    limiter.last_request_at = loop.time()


def _retry_delay(
    response: httpx.Response,
    attempt: int,
    min_interval: float,
) -> float:
    """Read Retry-After as seconds or an HTTP date, then fall back to backoff."""
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            seconds = float(retry_after)
            if math.isfinite(seconds):
                return max(0.0, seconds)
        except (TypeError, ValueError, OverflowError):
            pass

        try:
            retry_at = parsedate_to_datetime(retry_after)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError, IndexError):
            pass

    delay = min(10.0, 0.25 * (2**attempt))
    if response.status_code == 429:
        delay = max(delay, min(10.0, min_interval * (2 ** (attempt + 1))))
    return delay


class AcademicAPIError(RuntimeError):
    """An upstream API request or response failed in a known way."""

    def __init__(
        self,
        source: AcademicSource,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        self.source = source
        self.status_code = status_code
        super().__init__(f"{source}: {message}")


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _element_text(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    return _clean_text("".join(element.itertext()))


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _year_from(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        match = re.search(r"\b(?:18|19|20|21)\d{2}\b", str(value))
        if match:
            return int(match.group(0))
    return None


def _normalize_doi(value: Any) -> str | None:
    doi = _clean_text(value)
    if not doi:
        return None
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    return doi.rstrip(".,; ") or None


def _search_terms(query: str, *, limit: int = 8) -> list[str]:
    """Return distinct content words for provider-specific query syntax."""
    terms: list[str] = []
    seen: set[str] = set()
    for term in re.findall(r"[a-z0-9]+", query.casefold()):
        if len(term) < 2 or term in _QUERY_STOP_WORDS or term in seen:
            continue
        terms.append(term)
        seen.add(term)
        if len(terms) >= limit:
            break
    return terms


def _arxiv_search_expression(query: str) -> str:
    """Build a valid, broad arXiv expression from the planner's plain text."""
    terms = _search_terms(query)
    if not terms:
        terms = re.findall(r"[a-z0-9]+", query.casefold())[:8]
    return " OR ".join(f"all:{term}" for term in terms)


def _pubmed_broad_query(query: str) -> str | None:
    """Build a recall-oriented fallback for PubMed's implicit AND searches."""
    terms = _search_terms(query)
    if len(terms) < 2:
        return None
    return " OR ".join(terms)


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, dict):
            value = value.get("name") or value.get("category")
        text = _clean_text(value)
        if text and text not in seen:
            output.append(text)
            seen.add(text)
    return output


def _external_id_map(values: Any) -> dict[str, str]:
    if not isinstance(values, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in values.items():
        normalized = _clean_text(value)
        if normalized and isinstance(value, (str, int)):
            result[str(key).strip().lower()] = normalized
    return result


class AcademicClientService:
    """Search supported scholarly APIs and normalize their paper records.

    Use the service as an async context manager so its pooled HTTP client is
    closed promptly. An injected client remains owned by the caller.
    """

    def __init__(
        self,
        s2_api_key: str | None = None,
        email: str | None = None,
        *,
        ncbi_api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float | httpx.Timeout = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = 2,
        user_agent: str = "IntelliProof Academic Search/1.0",
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be zero or greater")

        self.s2_api_key = s2_api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY")
        self.ncbi_api_key = ncbi_api_key or os.getenv("NCBI_API_KEY")
        self.email = email or os.getenv("NCBI_EMAIL")
        self.max_retries = max_retries
        self.user_agent = f"{user_agent} (mailto:{self.email})" if self.email else user_agent
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": self.user_agent},
        )

    async def __aenter__(self) -> AcademicClientService:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the internally created HTTP client, if this service owns it."""
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    @staticmethod
    def _validate_search(query: str, limit: int) -> str:
        normalized_query = _clean_text(query)
        if not normalized_query:
            raise ValueError("query must not be empty")
        if not 1 <= limit <= MAX_RESULTS_PER_PROVIDER:
            raise ValueError(f"limit must be between 1 and {MAX_RESULTS_PER_PROVIDER}")
        return normalized_query

    async def _get(
        self,
        source: AcademicSource,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        request_headers = {"User-Agent": self.user_agent}
        if headers:
            request_headers.update(headers)

        credential = None
        min_interval = 0.0
        if source == "arxiv":
            min_interval = ARXIV_MIN_REQUEST_INTERVAL_SECONDS
        elif source == "semantic_scholar":
            credential = request_headers.get("x-api-key")
            min_interval = SEMANTIC_SCHOLAR_MIN_REQUEST_INTERVAL_SECONDS
        elif source == "pubmed":
            credential = str(params.get("api_key") or "") or None
            min_interval = (
                PUBMED_MIN_REQUEST_INTERVAL_WITH_KEY_SECONDS
                if credential
                else PUBMED_MIN_REQUEST_INTERVAL_NO_KEY_SECONDS
            )

        limiter = _get_provider_rate_limiter(source, credential)
        for attempt in range(self.max_retries + 1):
            should_retry = False
            async with limiter.lock:
                await _wait_for_provider_request_slot(limiter, min_interval)
                try:
                    response = await self._client.get(
                        url,
                        params=params,
                        headers=request_headers,
                    )
                except httpx.RequestError as exc:
                    if attempt >= self.max_retries:
                        raise AcademicAPIError(
                            source,
                            f"request failed ({exc.__class__.__name__})",
                        ) from exc
                    delay = 0.25 * (2**attempt)
                    limiter.blocked_until = max(
                        limiter.blocked_until,
                        asyncio.get_running_loop().time() + delay,
                    )
                    should_retry = True
                else:
                    if response.status_code in _RETRYABLE_STATUS_CODES:
                        delay = _retry_delay(response, attempt, min_interval)
                        limiter.blocked_until = max(
                            limiter.blocked_until,
                            asyncio.get_running_loop().time() + delay,
                        )
                        if attempt < self.max_retries:
                            should_retry = True
                        else:
                            raise AcademicAPIError(
                                source,
                                f"request returned HTTP {response.status_code} "
                                f"after {attempt + 1} attempt(s)",
                                status_code=response.status_code,
                            )
                    elif not response.is_success:
                        raise AcademicAPIError(
                            source,
                            f"request returned HTTP {response.status_code}",
                            status_code=response.status_code,
                        )
                    else:
                        return response

            if should_retry:
                continue

        # The loop always returns or raises; this guard keeps the return type
        # clear if retry behavior is changed in the future.
        raise AcademicAPIError(source, "request failed after retries")

    async def search(
        self,
        source: AcademicSource,
        query: str,
        limit: int = 5,
    ) -> list[NormalizedPaper]:
        """Dispatch one query to the requested provider."""
        searchers = {
            "arxiv": self.search_arxiv,
            "semantic_scholar": self.search_semantic_scholar,
            "pubmed": self.search_pubmed,
        }
        try:
            searcher = searchers[source]
        except KeyError as exc:
            raise ValueError(f"unsupported academic source: {source}") from exc
        return await searcher(query, limit)

    async def search_all(
        self,
        query: str,
        limit_per_source: int = 5,
        *,
        sources: Sequence[AcademicSource] | None = None,
    ) -> AcademicSearchResponse:
        """Search providers concurrently and preserve successful partial results."""
        normalized_query = self._validate_search(query, limit_per_source)
        selected_sources = tuple(sources or ("arxiv", "semantic_scholar", "pubmed"))
        if not selected_sources:
            return AcademicSearchResponse(query=normalized_query)

        searchers = {
            "arxiv": self.search_arxiv,
            "semantic_scholar": self.search_semantic_scholar,
            "pubmed": self.search_pubmed,
        }
        invalid_sources = [source for source in selected_sources if source not in searchers]
        if invalid_sources:
            raise ValueError(f"unsupported academic source(s): {', '.join(invalid_sources)}")

        outcomes = await asyncio.gather(
            *(searchers[source](normalized_query, limit_per_source) for source in selected_sources),
            return_exceptions=True,
        )
        papers: list[NormalizedPaper] = []
        errors: dict[AcademicSource, str] = {}
        for source, outcome in zip(selected_sources, outcomes, strict=True):
            if isinstance(outcome, AcademicAPIError):
                errors[source] = str(outcome)
            elif isinstance(outcome, BaseException):
                raise outcome
            else:
                papers.extend(outcome)
        return AcademicSearchResponse(query=normalized_query, papers=papers, errors=errors)

    async def search_semantic_scholar(
        self,
        query: str,
        limit: int = 5,
    ) -> list[NormalizedPaper]:
        """Search Semantic Scholar's Graph API."""
        normalized_query = self._validate_search(query, limit)
        params = {
            "query": normalized_query,
            "limit": limit,
            "fields": ",".join(
                (
                    "paperId",
                    "title",
                    "abstract",
                    "authors",
                    "year",
                    "publicationDate",
                    "citationCount",
                    "openAccessPdf",
                    "externalIds",
                    "url",
                    "venue",
                    "journal",
                    "isOpenAccess",
                    "fieldsOfStudy",
                    "publicationTypes",
                )
            ),
        }
        headers = {"x-api-key": self.s2_api_key} if self.s2_api_key else None
        response = await self._get(
            "semantic_scholar",
            SEMANTIC_SCHOLAR_API_URL,
            params=params,
            headers=headers,
        )
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise AcademicAPIError("semantic_scholar", "returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise AcademicAPIError("semantic_scholar", "returned an unexpected response")

        records = payload.get("data")
        if not isinstance(records, list):
            raise AcademicAPIError("semantic_scholar", "response did not contain a paper list")

        papers: list[NormalizedPaper] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            paper_id = _clean_text(record.get("paperId"))
            title = _clean_text(record.get("title"))
            if not paper_id or not title:
                continue

            external_ids = _external_id_map(record.get("externalIds"))
            external_ids["semantic_scholar"] = paper_id
            doi = _normalize_doi(external_ids.get("doi"))
            if doi:
                external_ids["doi"] = doi

            open_access_pdf = record.get("openAccessPdf")
            pdf_url = _clean_text(open_access_pdf.get("url")) if isinstance(open_access_pdf, dict) else None
            journal_data = record.get("journal")
            journal = _clean_text(journal_data.get("name")) if isinstance(journal_data, dict) else None
            url = _clean_text(record.get("url"))
            if not url:
                url = f"https://doi.org/{quote(doi, safe='/')}" if doi else (
                    f"https://www.semanticscholar.org/paper/{quote(paper_id, safe='')}"
                )

            is_open_access = record.get("isOpenAccess")
            papers.append(
                NormalizedPaper(
                    id=f"s2:{paper_id}",
                    source="semantic_scholar",
                    title=title,
                    abstract=_clean_text(record.get("abstract")),
                    authors=_string_list(record.get("authors")),
                    year=_as_int(record.get("year")),
                    publication_date=_clean_text(record.get("publicationDate")),
                    doi=doi,
                    url=url,
                    pdf_url=pdf_url,
                    citation_count=_as_int(record.get("citationCount")),
                    venue=_clean_text(record.get("venue")) or journal,
                    journal=journal,
                    is_open_access=is_open_access if isinstance(is_open_access, bool) else None,
                    fields_of_study=_string_list(record.get("fieldsOfStudy")),
                    publication_types=_string_list(record.get("publicationTypes")),
                    external_ids=external_ids,
                )
            )
        return papers

    async def search_arxiv(self, query: str, limit: int = 5) -> list[NormalizedPaper]:
        """Search arXiv and parse its Atom feed."""
        normalized_query = self._validate_search(query, limit)
        response = await self._get(
            "arxiv",
            ARXIV_API_URL,
            params={
                "search_query": _arxiv_search_expression(normalized_query),
                "start": 0,
                "max_results": limit,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
        )
        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as exc:
            raise AcademicAPIError("arxiv", "returned invalid Atom XML") from exc

        entries = root.findall(f"{{{_ATOM_NS}}}entry")
        if len(entries) == 1:
            entry = entries[0]
            error_title = _element_text(entry.find(f"{{{_ATOM_NS}}}title"))
            error_id = _element_text(entry.find(f"{{{_ATOM_NS}}}id")) or ""
            if error_title and error_title.casefold() == "error":
                error_detail = _element_text(entry.find(f"{{{_ATOM_NS}}}summary"))
                raise AcademicAPIError(
                    "arxiv",
                    f"search endpoint error: {error_detail or error_id or 'invalid query'}",
                )

        papers: list[NormalizedPaper] = []
        for entry in entries:
            paper_url = _element_text(entry.find(f"{{{_ATOM_NS}}}id"))
            title = _element_text(entry.find(f"{{{_ATOM_NS}}}title"))
            if not paper_url or not title:
                continue

            parsed_id = urlparse(paper_url).path.rstrip("/").rsplit("/", 1)[-1]
            if not parsed_id:
                continue
            version_match = re.fullmatch(r"(.+?)(v\d+)", parsed_id)
            canonical_id = version_match.group(1) if version_match else parsed_id
            canonical_url = f"https://arxiv.org/abs/{canonical_id}"

            pdf_url = None
            for link in entry.findall(f"{{{_ATOM_NS}}}link"):
                if link.get("title", "").lower() == "pdf" or link.get("type") == "application/pdf":
                    pdf_url = _clean_text(link.get("href"))
                    if pdf_url:
                        break
            if not pdf_url:
                pdf_url = f"https://arxiv.org/pdf/{canonical_id}"

            published = _element_text(entry.find(f"{{{_ATOM_NS}}}published"))
            authors = [
                name
                for author in entry.findall(f"{{{_ATOM_NS}}}author")
                if (name := _element_text(author.find(f"{{{_ATOM_NS}}}name")))
            ]
            doi = _normalize_doi(_element_text(entry.find(f"{{{_ARXIV_NS}}}doi")))
            external_ids = {"arxiv": canonical_id}
            if version_match:
                external_ids["arxiv_version"] = parsed_id
            if doi:
                external_ids["doi"] = doi

            papers.append(
                NormalizedPaper(
                    id=f"arxiv:{canonical_id}",
                    source="arxiv",
                    title=title,
                    abstract=_element_text(entry.find(f"{{{_ATOM_NS}}}summary")),
                    authors=authors,
                    year=_year_from(published),
                    publication_date=published,
                    doi=doi,
                    url=canonical_url,
                    pdf_url=pdf_url,
                    venue=_element_text(entry.find(f"{{{_ARXIV_NS}}}journal_ref")),
                    is_open_access=True,
                    fields_of_study=[
                        category.get("term", "").strip()
                        for category in entry.findall(f"{{{_ATOM_NS}}}category")
                        if category.get("term", "").strip()
                    ],
                    external_ids=external_ids,
                )
            )
        return papers

    def _pubmed_params(self) -> dict[str, str]:
        params = {"db": "pubmed", "tool": "IntelliProof"}
        if self.email:
            params["email"] = self.email
        if self.ncbi_api_key:
            params["api_key"] = self.ncbi_api_key
        return params

    async def search_pubmed(self, query: str, limit: int = 5) -> list[NormalizedPaper]:
        """Search PubMed and fetch citations, broadening only empty searches.

        PubMed's automatic term mapping ANDs untagged concepts together. The
        planner produces natural-language queries, so a long precise query can
        be too restrictive. In that case retry once with its distinct content
        terms joined by OR, then let Auto-Explore's graph matcher rank papers.
        """
        normalized_query = self._validate_search(query, limit)
        queries = [normalized_query]
        broad_query = _pubmed_broad_query(normalized_query)
        if broad_query and broad_query.casefold() != normalized_query.casefold():
            queries.append(broad_query)

        for candidate_query in queries:
            search_params: dict[str, Any] = {
                **self._pubmed_params(),
                "term": candidate_query,
                "retmode": "json",
                "retmax": limit,
            }
            search_response = await self._get(
                "pubmed",
                PUBMED_ESEARCH_URL,
                params=search_params,
            )
            try:
                search_payload = search_response.json()
            except (json.JSONDecodeError, ValueError) as exc:
                raise AcademicAPIError("pubmed", "search endpoint returned invalid JSON") from exc
            if not isinstance(search_payload, dict):
                raise AcademicAPIError("pubmed", "search endpoint returned an unexpected response")

            search_result = search_payload.get("esearchresult")
            id_list = search_result.get("idlist", []) if isinstance(search_result, dict) else []
            if not isinstance(id_list, list):
                raise AcademicAPIError("pubmed", "search response did not contain an identifier list")
            pmids = [str(pmid).strip() for pmid in id_list if str(pmid).strip()]
            if not pmids:
                continue

            fetch_params = {
                **self._pubmed_params(),
                "id": ",".join(pmids),
                "rettype": "abstract",
                "retmode": "xml",
            }
            fetch_response = await self._get("pubmed", PUBMED_EFETCH_URL, params=fetch_params)
            try:
                root = ET.fromstring(fetch_response.content)
            except ET.ParseError as exc:
                raise AcademicAPIError("pubmed", "fetch endpoint returned invalid XML") from exc

            papers_by_pmid: dict[str, NormalizedPaper] = {}
            for record in root.findall(".//PubmedArticle"):
                paper = self._parse_pubmed_article(record)
                if paper:
                    papers_by_pmid[paper.external_ids.get("pmid", "")] = paper
            # EFetch normally preserves search order; keeping ESearch order gives
            # callers consistent relevance ordering even if the feed does not.
            papers = [papers_by_pmid[pmid] for pmid in pmids if pmid in papers_by_pmid]
            if papers:
                return papers

        return []

    @staticmethod
    def _parse_pubmed_article(record: ET.Element) -> NormalizedPaper | None:
        citation = record.find("./MedlineCitation")
        article = citation.find("./Article") if citation is not None else None
        if citation is None or article is None:
            return None

        pmid = _element_text(citation.find("./PMID"))
        title = _element_text(article.find("./ArticleTitle"))
        if not pmid or not title:
            return None

        author_names: list[str] = []
        for author in article.findall("./AuthorList/Author"):
            name = _element_text(author.find("./CollectiveName"))
            if not name:
                fore_name = _element_text(author.find("./ForeName"))
                initials = _element_text(author.find("./Initials"))
                last_name = _element_text(author.find("./LastName"))
                given_name = fore_name or initials
                name = " ".join(part for part in (given_name, last_name) if part) or None
            if name and name not in author_names:
                author_names.append(name)

        abstract_sections: list[str] = []
        for section in article.findall("./Abstract/AbstractText"):
            text = _element_text(section)
            if not text:
                continue
            label = _clean_text(section.get("Label") or section.get("NlmCategory"))
            abstract_sections.append(f"{label}: {text}" if label else text)
        abstract = "\n\n".join(abstract_sections) or None

        pub_date = article.find("./Journal/JournalIssue/PubDate")
        date_parts = [_element_text(child) for child in list(pub_date)] if pub_date is not None else []
        publication_date = " ".join(part for part in date_parts if part) or None
        year = _year_from(publication_date)
        journal = _element_text(article.find("./Journal/Title"))

        external_ids = {"pmid": pmid}
        doi = None
        for article_id in record.findall("./PubmedData/ArticleIdList/ArticleId"):
            id_type = (article_id.get("IdType") or "").strip().lower()
            identifier = _element_text(article_id)
            if not id_type or not identifier:
                continue
            external_ids[id_type] = identifier
            if id_type == "doi":
                doi = _normalize_doi(identifier)
        if not doi:
            for electronic_id in article.findall("./ELocationID"):
                if (electronic_id.get("EIdType") or "").lower() == "doi":
                    doi = _normalize_doi(_element_text(electronic_id))
                    if doi:
                        external_ids["doi"] = doi
                        break

        publication_types = [
            value
            for item in article.findall("./PublicationTypeList/PublicationType")
            if (value := _element_text(item))
        ]
        return NormalizedPaper(
            id=f"pmid:{pmid}",
            source="pubmed",
            title=title,
            abstract=abstract,
            authors=author_names,
            year=year,
            publication_date=publication_date,
            doi=doi,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{quote(pmid, safe='')}/",
            venue=journal,
            journal=journal,
            publication_types=publication_types,
            external_ids=external_ids,
        )


# Keep the original prototype name available while making the module's role
# discoverable under the parser-oriented name used by new integrations.
AcademicAPIParser = AcademicClientService
