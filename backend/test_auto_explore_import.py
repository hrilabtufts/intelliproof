import asyncio
import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import ai_api
from ai_models import (
    AutoExploreClaimMatch,
    AutoExploreImportItem,
    AutoExplorePaperMatch,
)
from auto_explore_import import (
    download_open_access_pdf,
    enqueue_imported_documents,
    import_selected_papers,
)
from schemas.academic_api_schema import NormalizedPaper


@pytest.fixture(autouse=True)
def run_test_workers_inline(monkeypatch):
    async def inline_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", inline_to_thread)


class FakeQuery:
    def __init__(self, supabase, table):
        self.supabase = supabase
        self.name = table
        self.mode = "select"
        self.payload = None
        self.filters = {}

    def select(self, _columns):
        self.mode = "select"
        return self

    def insert(self, payload):
        self.mode = "insert"
        self.payload = payload
        return self

    def eq(self, key, value):
        self.filters[key] = value
        return self

    def limit(self, _count):
        return self

    def execute(self):
        if self.name == "supporting_documents" and self.mode == "select":
            documents = [
                item
                for item in self.supabase.documents
                if item.get("graph_id") == self.filters.get("graph_id")
            ]
            return SimpleNamespace(data=documents, error=None)
        if self.name == "supporting_documents" and self.mode == "insert":
            row = {**self.payload, "id": f"doc-{len(self.supabase.documents) + 1}"}
            self.supabase.documents.append(row)
            return SimpleNamespace(data=[row], error=None)
        if self.name == "profiles":
            return SimpleNamespace(data=[{"email": "owner@example.org"}], error=None)
        if self.name == "graphs":
            matches = self.filters.get("owner_email") == "owner@example.org"
            rows = [{"id": "graph-1"}] if matches else []
            return SimpleNamespace(data=rows, error=None)
        return SimpleNamespace(data=[], error=None)


class FakeBucket:
    def __init__(self, supabase):
        self.supabase = supabase

    def upload(self, path, content, _options):
        if self.supabase.fail_pdf_upload and path.endswith(".pdf"):
            raise RuntimeError("PDF storage limit exceeded")
        if self.supabase.fail_path_contains and self.supabase.fail_path_contains in path:
            raise RuntimeError("storage unavailable")
        self.supabase.files[path] = content

    def get_public_url(self, path):
        return f"https://storage.example/{path}"

    def remove(self, paths):
        self.supabase.removed.extend(paths)
        for path in paths:
            self.supabase.files.pop(path, None)


class FakeSupabase:
    def __init__(self, documents=None, fail_path_contains=None, fail_pdf_upload=False):
        self.documents = list(documents or [])
        self.files = {}
        self.removed = []
        self.fail_path_contains = fail_path_contains
        self.fail_pdf_upload = fail_pdf_upload
        self.storage = SimpleNamespace(from_=lambda _bucket: FakeBucket(self))
        self.auth = SimpleNamespace(
            get_user=lambda _token: SimpleNamespace(user=SimpleNamespace(id="user-1"))
        )

    def table(self, name):
        return FakeQuery(self, name)


def _match(paper_id="arxiv:123", *, title="Study of this claim", pdf_url=None, abstract="Abstract text."):
    paper = NormalizedPaper(
        id=paper_id,
        source="arxiv",
        title=title,
        abstract=abstract,
        authors=["A. Author"],
        year=2024,
        url=f"https://arxiv.org/abs/{paper_id.split(':')[-1]}",
        pdf_url=pdf_url,
        doi=f"10.1234/{paper_id.replace(':', '-')}",
        is_open_access=bool(pdf_url),
        external_ids={"arxiv": paper_id.split(":")[-1]},
    )
    return AutoExplorePaperMatch(
        paper=paper,
        claim_matches=[
            AutoExploreClaimMatch(
                claim_id="claim-1",
                relation="supports",
                rationale="The paper studies this claim.",
                confidence=0.8,
            )
        ],
    )


def test_import_stores_provider_pdf_and_match_provenance():
    supabase = FakeSupabase()
    match = _match(pdf_url="https://arxiv.org/pdf/123")

    result = asyncio.run(
        import_selected_papers(
            supabase=supabase,
            graph_id="graph-1",
            user_id="user-1",
            uploader_email="owner@example.org",
            selected_matches=[match],
            downloader=lambda _url: asyncio.sleep(0, result=b"%PDF-1.7 test"),
        )
    )

    assert result.results[0].status == "imported"
    assert result.results[0].storage_format == "pdf"
    assert result.results[0].indexing_status == "pending"
    saved = supabase.documents[0]
    assert saved["graph_id"] == "graph-1"
    assert saved["metadata"]["academic_id"] == "arxiv:123"
    assert saved["metadata"]["claim_matches"][0]["claim_id"] == "claim-1"
    assert next(iter(supabase.files.values())).startswith(b"%PDF-")


def test_import_falls_back_to_text_when_pdf_is_unavailable():
    supabase = FakeSupabase()
    match = _match(pdf_url="https://arxiv.org/pdf/123")

    async def unavailable(_url):
        raise RuntimeError("provider denied the PDF request")

    result = asyncio.run(
        import_selected_papers(
            supabase=supabase,
            graph_id="graph-1",
            user_id="user-1",
            uploader_email="owner@example.org",
            selected_matches=[match],
            downloader=unavailable,
        )
    )

    assert result.results[0].status == "imported"
    assert result.results[0].storage_format == "text"
    assert "provider denied" in result.results[0].fallback_reason
    text = next(iter(supabase.files.values())).decode("utf-8")
    assert "Abstract text." in text
    assert "Paper link: https://arxiv.org/abs/123" in text


def test_import_falls_back_to_text_when_pdf_storage_fails():
    supabase = FakeSupabase(fail_pdf_upload=True)
    match = _match(pdf_url="https://arxiv.org/pdf/123")

    result = asyncio.run(
        import_selected_papers(
            supabase=supabase,
            graph_id="graph-1",
            user_id="user-1",
            uploader_email="owner@example.org",
            selected_matches=[match],
            downloader=lambda _url: asyncio.sleep(0, result=b"%PDF-1.7 test"),
        )
    )

    assert result.results[0].status == "imported"
    assert result.results[0].storage_format == "text"
    assert "PDF could not be stored" in result.results[0].fallback_reason
    assert supabase.documents[0]["metadata"]["storage_format"] == "text"


def test_text_fallback_still_contains_citation_and_link_without_abstract():
    supabase = FakeSupabase()
    match = _match(abstract=None)

    result = asyncio.run(
        import_selected_papers(
            supabase=supabase,
            graph_id="graph-1",
            user_id="user-1",
            uploader_email="owner@example.org",
            selected_matches=[match],
        )
    )

    assert result.results[0].storage_format == "text"
    text = next(iter(supabase.files.values())).decode("utf-8")
    assert "Study of this claim" in text
    assert "Abstract unavailable." in text
    assert "Paper link: https://arxiv.org/abs/123" in text


def test_import_rechecks_duplicates_before_download_or_storage():
    supabase = FakeSupabase(
        documents=[
            {
                "id": "existing-doc",
                "graph_id": "graph-1",
                "name": "existing.pdf",
                "metadata": {
                    "academic_id": "arxiv:123",
                    "title": "Study of this claim",
                },
            }
        ]
    )

    def should_not_download(_url):
        raise AssertionError("duplicate papers must not be downloaded")

    result = asyncio.run(
        import_selected_papers(
            supabase=supabase,
            graph_id="graph-1",
            user_id="user-1",
            uploader_email="owner@example.org",
            selected_matches=[_match(pdf_url="https://arxiv.org/pdf/123")],
            downloader=should_not_download,
        )
    )

    assert result.results[0].status == "duplicate"
    assert result.results[0].document_id == "existing-doc"
    assert supabase.files == {}


def test_import_deduplicates_against_an_existing_document_filename():
    supabase = FakeSupabase(
        documents=[
            {
                "id": "existing-file",
                "graph_id": "graph-1",
                "name": "Study of this claim.pdf",
                "metadata": {},
            }
        ]
    )

    result = asyncio.run(
        import_selected_papers(
            supabase=supabase,
            graph_id="graph-1",
            user_id="user-1",
            uploader_email="owner@example.org",
            selected_matches=[_match()],
        )
    )

    assert result.results[0].status == "duplicate"
    assert result.results[0].document_id == "existing-file"


def test_import_queues_only_successfully_stored_documents_for_indexing():
    queued = []

    class BackgroundTasks:
        def add_task(self, *args):
            queued.append(args)

    def ingestion_task(*_args):
        pass

    def request_factory(**kwargs):
        return kwargs

    count = enqueue_imported_documents(
        results=[
            AutoExploreImportItem(
                paper_id="arxiv:1",
                status="imported",
                document_id="doc-1",
                document_name="paper.pdf",
                document_url="https://storage.example/paper.pdf",
                document_type="document",
            ),
            AutoExploreImportItem(paper_id="arxiv:2", status="duplicate"),
            AutoExploreImportItem(paper_id="arxiv:3", status="failed"),
        ],
        background_tasks=BackgroundTasks(),
        user_id="user-1",
        ingestion_task=ingestion_task,
        ingestion_request_factory=request_factory,
    )

    assert count == 1
    assert len(queued) == 1
    assert queued[0][0] is ingestion_task
    assert queued[0][1]["document_id"] == "doc-1"
    assert queued[0][2] == "user-1"


def test_import_reports_partial_failures_and_stores_successes():
    supabase = FakeSupabase(fail_path_contains="Storage_failure")
    first = _match("arxiv:success", title="Successful study")
    second = _match("arxiv:failure", title="Storage failure")

    result = asyncio.run(
        import_selected_papers(
            supabase=supabase,
            graph_id="graph-1",
            user_id="user-1",
            uploader_email="owner@example.org",
            selected_matches=[first, second],
        )
    )

    assert [item.status for item in result.results] == ["imported", "failed"]
    assert len(supabase.documents) == 1
    assert "storage unavailable" in result.results[1].error


def test_pdf_downloader_rejects_non_https_and_private_hosts():
    with pytest.raises(Exception, match="public HTTPS"):
        asyncio.run(download_open_access_pdf("http://127.0.0.1/paper.pdf"))


def test_graph_access_requires_session_and_owner(monkeypatch):
    with pytest.raises(HTTPException) as unauthenticated:
        ai_api._require_auto_explore_graph_access("graph-1", None)
    assert unauthenticated.value.status_code == 401

    supabase = FakeSupabase()
    monkeypatch.setattr(ai_api, "_rag_supabase_client", lambda: supabase)
    identity = ai_api._require_auto_explore_graph_access(
        "graph-1", "Bearer valid-session"
    )
    assert identity == ("user-1", "owner@example.org")

    class NonOwnerSupabase(FakeSupabase):
        def table(self, name):
            query = FakeQuery(self, name)
            if name == "profiles":
                query.execute = lambda: SimpleNamespace(
                    data=[{"email": "other@example.org"}], error=None
                )
            return query

    monkeypatch.setattr(ai_api, "_rag_supabase_client", NonOwnerSupabase)
    with pytest.raises(HTTPException) as forbidden:
        ai_api._require_auto_explore_graph_access(
            "graph-1", "Bearer valid-session"
        )
    assert forbidden.value.status_code == 404
