import asyncio
import json

import pytest

from academic_api_parser import AcademicAPIError
from auto_explorer import run_auto_explorer
from schemas.academic_api_schema import NormalizedPaper


def _paper(paper_id, *, source="arxiv", doi=None, title=None, pdf_url=None):
    return NormalizedPaper(
        id=paper_id,
        source=source,
        title=title or f"Research {paper_id}",
        abstract="A short abstract about the claim.",
        url=f"https://example.org/{paper_id}",
        pdf_url=pdf_url,
        doi=doi,
        external_ids={source: paper_id.split(":", 1)[-1]},
    )


class FakeAcademicClient:
    def __init__(self, records):
        self.records = records
        self.calls = []

    async def search(self, source, query, limit):
        self.calls.append((source, query, limit))
        outcome = self.records.get(source, [])
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


GRAPH = {
    "nodes": [{"id": "claim-1", "data": {"text": "The claim under review"}}],
    "edges": [{"id": "edge-1", "source": "claim-1", "target": "claim-2", "data": {"relation": "supporting"}}],
    "evidence": [],
    "supportingDocuments": [],
}


@pytest.fixture(autouse=True)
def run_test_workers_inline(monkeypatch):
    async def inline_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", inline_to_thread)


def test_search_uses_only_selected_sources_and_validates_claim_matches():
    paper = _paper("arxiv:123", title="Research about the claim")
    client = FakeAcademicClient({"arxiv": [paper], "pubmed": [_paper("pmid:999", source="pubmed")]})
    matcher = lambda _: json.dumps(
        {
            "matches": [
                {
                    "paper_id": "arxiv:123",
                    "claim_matches": [
                        {
                            "claim_id": "claim-1",
                            "relation": "supports",
                            "rationale": "The abstract studies the claim.",
                            "confidence": 0.8,
                        },
                        {
                            "claim_id": "not-in-graph",
                            "relation": "supports",
                            "rationale": "Untrusted model output.",
                            "confidence": 0.99,
                        },
                    ],
                }
            ]
        }
    )

    result = asyncio.run(
        run_auto_explorer(
            graph_data=GRAPH,
            sources=["arxiv"],
            academic_client=client,
            planner=lambda _: '{"queries":["claim research"]}',
            matcher=matcher,
        )
    )

    assert [call[0] for call in client.calls] == ["arxiv"]
    assert [match.paper.id for match in result.matches] == ["arxiv:123"]
    assert [match.claim_id for match in result.matches[0].claim_matches] == ["claim-1"]


def test_search_keeps_successful_sources_when_another_provider_fails():
    paper = _paper("pmid:123", source="pubmed", title="Relevant clinical study")
    client = FakeAcademicClient(
        {
            "arxiv": AcademicAPIError("arxiv", "temporarily unavailable"),
            "pubmed": [paper],
        }
    )
    matcher = lambda _: json.dumps(
        {
            "matches": [
                {
                    "paper_id": "pmid:123",
                    "claim_matches": [
                        {
                            "claim_id": "claim-1",
                            "relation": "contextualizes",
                            "rationale": "The study covers this claim area.",
                            "confidence": 0.7,
                        }
                    ],
                }
            ]
        }
    )

    result = asyncio.run(
        run_auto_explorer(
            graph_data=GRAPH,
            sources=["arxiv", "pubmed"],
            academic_client=client,
            planner=lambda _: '{"queries":["clinical study"]}',
            matcher=matcher,
        )
    )

    assert {call[0] for call in client.calls} == {"arxiv", "pubmed"}
    assert [match.paper.id for match in result.matches] == ["pmid:123"]
    assert "0/1 searches succeeded" in result.errors["arxiv"]
    assert "temporarily unavailable" in result.errors["arxiv"]


def test_search_reports_partial_provider_failure_without_discarding_papers():
    paper = _paper("s2:123", source="semantic_scholar", title="Relevant study")

    class PartialSemanticScholarClient:
        async def search(self, source, query, limit):
            if query == "second query":
                raise AcademicAPIError(
                    source,
                    "request returned HTTP 429 after 3 attempt(s)",
                    status_code=429,
                )
            return [paper]

    matcher = lambda _: json.dumps(
        {
            "matches": [
                {
                    "paper_id": "s2:123",
                    "claim_matches": [
                        {
                            "claim_id": "claim-1",
                            "relation": "contextualizes",
                            "rationale": "The study addresses the graph claim.",
                            "confidence": 0.8,
                        }
                    ],
                }
            ]
        }
    )

    result = asyncio.run(
        run_auto_explorer(
            graph_data=GRAPH,
            sources=["semantic_scholar"],
            academic_client=PartialSemanticScholarClient(),
            planner=lambda _: '{"queries":["first query","second query"]}',
            matcher=matcher,
        )
    )

    assert [match.paper.id for match in result.matches] == ["s2:123"]
    assert "semantic_scholar: 1" in result.summary
    assert "1/2 searches succeeded" in result.errors["semantic_scholar"]
    assert "1 paper record(s) returned" in result.errors["semantic_scholar"]
    assert '"second query": request returned HTTP 429 after 3 attempt(s)' in result.errors[
        "semantic_scholar"
    ]


def test_search_filters_matches_that_duplicate_existing_documents():
    existing = {
        "nodes": GRAPH["nodes"],
        "edges": GRAPH["edges"],
        "evidence": [],
        "supportingDocuments": [
            {
                "id": "doc-1",
                "name": "Existing study.pdf",
                "metadata": {"title": "Existing study", "doi": "10.1234/existing"},
            }
        ],
    }
    duplicate = _paper(
        "arxiv:old",
        doi="10.1234/existing",
        title="Existing study",
    )
    fresh = _paper("arxiv:new", title="A fresh study")
    client = FakeAcademicClient({"arxiv": [duplicate, fresh]})
    seen_candidate_payload = []

    def matcher(prompt):
        seen_candidate_payload.append(prompt)
        return json.dumps(
            {
                "matches": [
                    {
                        "paper_id": "arxiv:new",
                        "claim_matches": [
                            {
                                "claim_id": "claim-1",
                                "relation": "challenges",
                                "rationale": "The paper tests the same claim.",
                                "confidence": 0.82,
                            }
                        ],
                    }
                ]
            }
        )

    result = asyncio.run(
        run_auto_explorer(
            graph_data=existing,
            sources=["arxiv"],
            academic_client=client,
            planner=lambda _: '{"queries":["study"]}',
            matcher=matcher,
        )
    )

    assert len(seen_candidate_payload) == 1
    assert "arxiv:old" not in seen_candidate_payload[0]
    assert [match.paper.id for match in result.matches] == ["arxiv:new"]
