import asyncio

import httpx
import pytest

import academic_api_parser as academic_api


@pytest.mark.parametrize(
    ("source", "interval_setting", "with_ncbi_key"),
    [
        ("arxiv", "ARXIV_MIN_REQUEST_INTERVAL_SECONDS", False),
        (
            "semantic_scholar",
            "SEMANTIC_SCHOLAR_MIN_REQUEST_INTERVAL_SECONDS",
            False,
        ),
        ("pubmed", "PUBMED_MIN_REQUEST_INTERVAL_NO_KEY_SECONDS", False),
        ("pubmed", "PUBMED_MIN_REQUEST_INTERVAL_WITH_KEY_SECONDS", True),
    ],
)
def test_concurrent_provider_requests_are_paced(
    monkeypatch,
    source,
    interval_setting,
    with_ncbi_key,
):
    interval = 0.02
    monkeypatch.setattr(academic_api, interval_setting, interval)

    async def run():
        starts = []
        active_requests = 0
        max_active_requests = 0

        async def handler(request):
            nonlocal active_requests, max_active_requests
            starts.append(asyncio.get_running_loop().time())
            active_requests += 1
            max_active_requests = max(max_active_requests, active_requests)
            await asyncio.sleep(0.003)
            active_requests -= 1
            return httpx.Response(200, json={}, request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        service = academic_api.AcademicClientService(
            client=client,
            ncbi_api_key="test-ncbi-key" if with_ncbi_key else None,
            max_retries=0,
        )
        params = {"api_key": "test-ncbi-key"} if with_ncbi_key else {}
        try:
            await asyncio.gather(
                *(
                    service._get(
                        source,
                        "https://provider.example/search",
                        params=params,
                    )
                    for _ in range(3)
                )
            )
        finally:
            await client.aclose()

        assert len(starts) == 3
        assert all(
            later - earlier >= interval * 0.8
            for earlier, later in zip(starts, starts[1:])
        )
        # Serializing the requests also satisfies arXiv's single-connection rule.
        assert max_active_requests == 1

    asyncio.run(run())


def test_retry_after_pauses_all_queued_semantic_scholar_requests(monkeypatch):
    monkeypatch.setattr(
        academic_api,
        "SEMANTIC_SCHOLAR_MIN_REQUEST_INTERVAL_SECONDS",
        0.01,
    )

    async def run():
        starts = []
        first_throttled_response_at = None

        async def handler(request):
            nonlocal first_throttled_response_at
            starts.append(asyncio.get_running_loop().time())
            if len(starts) == 1:
                first_throttled_response_at = asyncio.get_running_loop().time()
                return httpx.Response(
                    429,
                    headers={"Retry-After": "0.05"},
                    json={"error": "rate limited"},
                    request=request,
                )
            return httpx.Response(200, json={}, request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        service = academic_api.AcademicClientService(client=client, max_retries=1)
        try:
            await asyncio.gather(
                service._get(
                    "semantic_scholar",
                    "https://provider.example/search",
                    params={"query": "first"},
                ),
                service._get(
                    "semantic_scholar",
                    "https://provider.example/search",
                    params={"query": "second"},
                ),
            )
        finally:
            await client.aclose()

        assert first_throttled_response_at is not None
        assert len(starts) == 3
        assert min(starts[1:]) - first_throttled_response_at >= 0.04

    asyncio.run(run())


def test_semantic_scholar_429_without_retry_after_uses_exponential_backoff(
    monkeypatch,
):
    monkeypatch.setattr(
        academic_api,
        "SEMANTIC_SCHOLAR_MIN_REQUEST_INTERVAL_SECONDS",
        0.01,
    )

    async def run():
        starts = []

        async def handler(request):
            starts.append(asyncio.get_running_loop().time())
            if len(starts) < 3:
                return httpx.Response(429, json={"error": "rate limited"}, request=request)
            return httpx.Response(200, json={}, request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        service = academic_api.AcademicClientService(client=client, max_retries=2)
        try:
            await service._get(
                "semantic_scholar",
                "https://provider.example/search",
                params={"query": "test"},
            )
        finally:
            await client.aclose()

        assert len(starts) == 3
        assert starts[1] - starts[0] >= 0.018
        assert starts[2] - starts[1] >= 0.038

    asyncio.run(run())


def test_arxiv_search_uses_fielded_terms_and_parses_paper(monkeypatch):
    monkeypatch.setattr(academic_api, "ARXIV_MIN_REQUEST_INTERVAL_SECONDS", 0.01)

    async def run():
        observed_queries = []

        async def handler(request):
            observed_queries.append(request.url.params["search_query"])
            return httpx.Response(
                200,
                text=(
                    '<feed xmlns="http://www.w3.org/2005/Atom">'
                    "<entry>"
                    "<id>https://arxiv.org/abs/2601.12345v2</id>"
                    "<title>Hypertensive emergency study</title>"
                    "<summary>A study abstract.</summary>"
                    "<published>2026-01-01T00:00:00Z</published>"
                    "</entry></feed>"
                ),
                request=request,
            )

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        service = academic_api.AcademicClientService(client=http_client, max_retries=0)
        try:
            papers = await service.search_arxiv(
                "hypertensive emergency retinal hemorrhages and acute organ damage"
            )
        finally:
            await http_client.aclose()

        assert observed_queries == [
            "all:hypertensive OR all:emergency OR all:retinal OR "
            "all:hemorrhages OR all:acute OR all:organ OR all:damage"
        ]
        assert [paper.id for paper in papers] == ["arxiv:2601.12345"]

    asyncio.run(run())


def test_arxiv_error_feed_is_reported_as_a_provider_error(monkeypatch):
    monkeypatch.setattr(academic_api, "ARXIV_MIN_REQUEST_INTERVAL_SECONDS", 0.01)

    async def run():
        async def handler(request):
            return httpx.Response(
                200,
                text=(
                    '<feed xmlns="http://www.w3.org/2005/Atom">'
                    "<entry><id>https://arxiv.org/api/errors#query</id>"
                    "<title>Error</title><summary>Invalid search expression</summary>"
                    "</entry></feed>"
                ),
                request=request,
            )

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        service = academic_api.AcademicClientService(client=http_client, max_retries=0)
        try:
            with pytest.raises(academic_api.AcademicAPIError, match="Invalid search expression"):
                await service.search_arxiv("hypertensive emergency")
        finally:
            await http_client.aclose()

    asyncio.run(run())


def test_pubmed_retries_empty_natural_language_search_with_broader_terms(monkeypatch):
    monkeypatch.setattr(academic_api, "PUBMED_MIN_REQUEST_INTERVAL_NO_KEY_SECONDS", 0.01)

    async def run():
        observed_terms = []

        async def handler(request):
            if request.url.path.endswith("esearch.fcgi"):
                term = request.url.params["term"]
                observed_terms.append(term)
                ids = [] if len(observed_terms) == 1 else ["12345"]
                return httpx.Response(
                    200,
                    json={"esearchresult": {"idlist": ids}},
                    request=request,
                )
            return httpx.Response(
                200,
                text=(
                    "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
                    "<PMID>12345</PMID><Article><ArticleTitle>Hypertensive emergency study"
                    "</ArticleTitle><Journal><JournalIssue><PubDate><Year>2025</Year>"
                    "</PubDate></JournalIssue><Title>Clinical Journal</Title></Journal>"
                    "<Abstract><AbstractText>Study abstract.</AbstractText></Abstract>"
                    "</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"
                ),
                request=request,
            )

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        service = academic_api.AcademicClientService(client=http_client, max_retries=0)
        try:
            papers = await service.search_pubmed(
                "hypertensive emergency retinal hemorrhages diagnostic specificity"
            )
        finally:
            await http_client.aclose()

        assert observed_terms == [
            "hypertensive emergency retinal hemorrhages diagnostic specificity",
            "hypertensive OR emergency OR retinal OR hemorrhages OR diagnostic OR specificity",
        ]
        assert [paper.id for paper in papers] == ["pmid:12345"]
        assert papers[0].abstract == "Study abstract."

    asyncio.run(run())
