"""Persist user-selected Auto-Explore papers as graph supporting documents."""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Callable
from urllib.parse import urljoin, urlsplit

import httpx

if __package__:
    from .ai_models import (
        AutoExploreImportItem,
        AutoExploreImportResponse,
        AutoExplorePaperMatch,
    )
else:  # Support importing backend modules directly from backend/.
    from ai_models import AutoExploreImportItem, AutoExploreImportResponse, AutoExplorePaperMatch


STORAGE_BUCKET = "supporting-document-storage"
MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
MAX_DOWNLOAD_REDIRECTS = 3
MAX_IMPORT_CONCURRENCY = 3
_DOI_PREFIX = re.compile(r"^https?://(?:dx\.)?doi\.org/|^doi:\s*", re.IGNORECASE)


class PaperDownloadError(RuntimeError):
    """A declared PDF could not be downloaded as a safe, usable PDF."""


def _normalize_doi(value: Any) -> str:
    if value is None:
        return ""
    return _DOI_PREFIX.sub("", str(value).strip()).rstrip(".,; ").casefold()


def _normalize_title(value: Any) -> str:
    title = re.sub(r"\.(?:pdf|txt|doc|docx|md)$", "", str(value or ""), flags=re.IGNORECASE)
    return " ".join(re.findall(r"[\w]+", title.casefold()))


def _paper_identity_keys(paper: Any) -> set[str]:
    keys = {f"paper:{str(paper.id).casefold()}"}
    doi = _normalize_doi(paper.doi or paper.external_ids.get("doi"))
    if doi:
        keys.add(f"doi:{doi}")
    title = _normalize_title(paper.title)
    if title:
        keys.add(f"title:{title}")
    for key, value in (paper.external_ids or {}).items():
        if value:
            keys.add(f"external:{str(key).casefold()}:{str(value).strip().casefold()}")
    return keys


def _document_identity_keys(document: Mapping[str, Any]) -> set[str]:
    metadata = document.get("metadata")
    if isinstance(metadata, str):
        try:
            import json

            metadata = json.loads(metadata)
        except (TypeError, ValueError):
            metadata = {}
    if not isinstance(metadata, Mapping):
        metadata = {}

    keys: set[str] = set()
    doi = _normalize_doi(
        document.get("doi")
        or metadata.get("doi")
        or metadata.get("paper_url")
    )
    if doi:
        keys.add(f"doi:{doi}")
    title = _normalize_title(
        metadata.get("title") or document.get("title") or document.get("name")
    )
    if title:
        keys.add(f"title:{title}")

    academic_id = metadata.get("academic_id") or metadata.get("paper_id")
    if academic_id:
        keys.add(f"paper:{str(academic_id).casefold()}")
    external_ids = metadata.get("external_ids") or metadata.get("externalIds") or {}
    if isinstance(external_ids, Mapping):
        for key, value in external_ids.items():
            if value:
                keys.add(f"external:{str(key).casefold()}:{str(value).strip().casefold()}")
    return keys


async def _assert_public_https_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise PaperDownloadError("PDF URL must use a public HTTPS address")
    if parsed.port not in (None, 443):
        raise PaperDownloadError("PDF URL uses an unsupported port")

    hostname = parsed.hostname.rstrip(".")
    try:
        try:
            literal_ip = ipaddress.ip_address(hostname)
            addresses = [literal_ip]
        except ValueError:
            resolved = await asyncio.to_thread(
                socket.getaddrinfo,
                hostname,
                443,
                type=socket.SOCK_STREAM,
            )
            addresses = [ipaddress.ip_address(item[4][0]) for item in resolved]
    except (OSError, ValueError) as error:
        raise PaperDownloadError("PDF host could not be resolved") from error

    if not addresses or any(not address.is_global for address in addresses):
        raise PaperDownloadError("PDF URL does not resolve to a public host")


async def download_open_access_pdf(
    url: str,
    *,
    max_bytes: int = MAX_DOCUMENT_BYTES,
    timeout_seconds: float = 20.0,
) -> bytes:
    """Download a provider-declared PDF with redirect, host, and size checks."""
    headers = {"User-Agent": "IntelliProof Auto-Explore/1.0"}
    current_url = url
    async with httpx.AsyncClient(
        timeout=timeout_seconds,
        follow_redirects=False,
        headers=headers,
    ) as client:
        for redirect_count in range(MAX_DOWNLOAD_REDIRECTS + 1):
            await _assert_public_https_url(current_url)
            try:
                async with client.stream("GET", current_url) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location or redirect_count >= MAX_DOWNLOAD_REDIRECTS:
                            raise PaperDownloadError("PDF redirect limit exceeded")
                        current_url = urljoin(current_url, location)
                        continue
                    if not response.is_success:
                        raise PaperDownloadError(
                            f"PDF provider returned HTTP {response.status_code}"
                        )

                    content_length = response.headers.get("content-length")
                    if content_length:
                        try:
                            if int(content_length) > max_bytes:
                                raise PaperDownloadError("PDF exceeds the storage size limit")
                        except ValueError:
                            pass

                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        content.extend(chunk)
                        if len(content) > max_bytes:
                            raise PaperDownloadError("PDF exceeds the storage size limit")

                    if not bytes(content).startswith(b"%PDF-"):
                        raise PaperDownloadError("Provider response is not a PDF file")
                    return bytes(content)
            except httpx.HTTPError as error:
                raise PaperDownloadError("PDF download failed") from error

    raise PaperDownloadError("PDF download failed")


def _paper_text(match: AutoExplorePaperMatch) -> str:
    paper = match.paper
    lines = [paper.title]
    if paper.authors:
        lines.append(f"Authors: {', '.join(paper.authors)}")
    if paper.year:
        lines.append(f"Year: {paper.year}")
    if paper.venue or paper.journal:
        lines.append(f"Publication: {paper.venue or paper.journal}")
    if paper.doi:
        lines.append(f"DOI: {paper.doi}")
    lines.append(f"Source: {paper.source}")
    lines.append(f"Paper link: {paper.url}")
    lines.append("")
    lines.append("Abstract:")
    lines.append(paper.abstract or "Abstract unavailable.")
    return "\n".join(lines) + "\n"


def _safe_file_stem(title: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("._-")
    return (stem or "academic-paper")[:160]


def _store_document(
    *,
    supabase: Any,
    graph_id: str,
    user_id: str,
    uploader_email: str,
    match: AutoExplorePaperMatch,
    content: bytes,
    storage_format: str,
    fallback_reason: str | None,
) -> dict[str, Any]:
    paper = match.paper
    extension = "pdf" if storage_format == "pdf" else "txt"
    content_type = "application/pdf" if storage_format == "pdf" else "text/plain; charset=utf-8"
    filename = f"{_safe_file_stem(paper.title)}.{extension}"
    storage_path = f"{user_id}/auto-explore/{uuid.uuid4().hex}-{filename}"
    storage = supabase.storage.from_(STORAGE_BUCKET)
    uploaded = False
    try:
        storage.upload(
            storage_path,
            content,
            {"content-type": content_type, "x-upsert": "false"},
        )
        uploaded = True
        storage_url = storage.get_public_url(storage_path)
        metadata = {
            "title": paper.title,
            "source": paper.source,
            "academic_id": paper.id,
            "paper_url": paper.url,
            "pdf_url": paper.pdf_url,
            "doi": paper.doi,
            "external_ids": paper.external_ids,
            "authors": paper.authors,
            "year": paper.year,
            "publication_date": paper.publication_date,
            "venue": paper.venue,
            "journal": paper.journal,
            "storage_format": storage_format,
            "fallback_reason": fallback_reason,
            "claim_matches": [item.model_dump(mode="json") for item in match.claim_matches],
        }
        response = (
            supabase.table("supporting_documents")
            .insert(
                {
                    "graph_id": graph_id,
                    "name": filename,
                    "type": "document",
                    "url": storage_url,
                    "size": len(content),
                    "uploader_email": uploader_email,
                    "metadata": metadata,
                }
            )
            .execute()
        )
        if getattr(response, "error", None):
            raise RuntimeError("Could not save supporting document metadata")
        row = next(iter(response.data or []), None)
        if not row:
            raise RuntimeError("Supporting document metadata was not returned")
        return row
    except Exception:
        if uploaded:
            try:
                storage.remove([storage_path])
            except Exception:
                pass
        raise


def _list_graph_documents(supabase: Any, graph_id: str) -> list[Mapping[str, Any]]:
    response = (
        supabase.table("supporting_documents")
        .select("id,name,url,metadata")
        .eq("graph_id", graph_id)
        .execute()
    )
    if getattr(response, "error", None):
        raise RuntimeError("Could not check existing graph documents")
    return [item for item in (response.data or []) if isinstance(item, Mapping)]


async def import_selected_papers(
    *,
    supabase: Any,
    graph_id: str,
    user_id: str,
    uploader_email: str,
    selected_matches: Sequence[AutoExplorePaperMatch],
    downloader=download_open_access_pdf,
) -> AutoExploreImportResponse:
    """Store selected, matched papers and report outcomes independently."""
    try:
        existing_documents = await asyncio.to_thread(
            _list_graph_documents, supabase, graph_id
        )
    except Exception as error:
        raise RuntimeError("Could not check existing graph documents") from error

    known_identities: dict[str, str] = {}
    for document in existing_documents:
        document_id = str(document.get("id") or "")
        for identity in _document_identity_keys(document):
            known_identities.setdefault(identity, document_id)

    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(MAX_IMPORT_CONCURRENCY)

    async def import_one(match: AutoExplorePaperMatch) -> AutoExploreImportItem:
        paper_id = match.paper.id
        identities = _paper_identity_keys(match.paper)
        async with lock:
            duplicate_ids = [known_identities[key] for key in identities if key in known_identities]
            if duplicate_ids:
                return AutoExploreImportItem(
                    paper_id=paper_id,
                    status="duplicate",
                    document_id=next((item for item in duplicate_ids if item), None),
                )
            for identity in identities:
                known_identities[identity] = ""

        async with semaphore:
            storage_format = "pdf"
            fallback_reason: str | None = None
            content: bytes | None = None
            if match.paper.pdf_url and match.paper.is_open_access is not False:
                try:
                    content = await downloader(match.paper.pdf_url)
                except Exception as error:
                    fallback_reason = str(error)[:240] or "PDF could not be downloaded"
            else:
                fallback_reason = "No provider-declared open-access PDF was available"

            if content is None:
                storage_format = "text"
                content = _paper_text(match).encode("utf-8")

            try:
                row = await asyncio.to_thread(
                    _store_document,
                    supabase=supabase,
                    graph_id=graph_id,
                    user_id=user_id,
                    uploader_email=uploader_email,
                    match=match,
                    content=content,
                    storage_format=storage_format,
                    fallback_reason=fallback_reason,
                )
            except Exception as error:
                if storage_format == "pdf":
                    fallback_reason = "PDF could not be stored; a text document was used"
                    storage_format = "text"
                    content = _paper_text(match).encode("utf-8")
                    try:
                        row = await asyncio.to_thread(
                            _store_document,
                            supabase=supabase,
                            graph_id=graph_id,
                            user_id=user_id,
                            uploader_email=uploader_email,
                            match=match,
                            content=content,
                            storage_format=storage_format,
                            fallback_reason=fallback_reason,
                        )
                    except Exception as fallback_error:
                        error = fallback_error
                        row = None
                else:
                    row = None

                if row is None:
                    async with lock:
                        for identity in identities:
                            if not known_identities.get(identity):
                                known_identities.pop(identity, None)
                    return AutoExploreImportItem(
                        paper_id=paper_id,
                        status="failed",
                        error=str(error)[:300] or "Document import failed",
                    )

            async with lock:
                document_id = str(row.get("id") or "")
                for identity in identities:
                    known_identities[identity] = document_id

            return AutoExploreImportItem(
                paper_id=paper_id,
                status="imported",
                storage_format=storage_format,
                fallback_reason=fallback_reason,
                document_id=str(row.get("id")) if row.get("id") else None,
                document_name=str(row.get("name") or "") or None,
                document_url=str(row.get("url") or "") or None,
                document_type=str(row.get("type") or "document"),
                indexing_status="pending",
            )

    results = await asyncio.gather(*(import_one(match) for match in selected_matches))
    imported = sum(item.status == "imported" for item in results)
    duplicates = sum(item.status == "duplicate" for item in results)
    failed = sum(item.status == "failed" for item in results)
    summary = f"Imported {imported} paper(s); {duplicates} already present; {failed} failed."
    return AutoExploreImportResponse(results=results, summary=summary)


def enqueue_imported_documents(
    *,
    results: Sequence[AutoExploreImportItem],
    background_tasks: Any,
    user_id: str,
    ingestion_task: Callable[..., Any],
    ingestion_request_factory: Callable[..., Any],
) -> int:
    """Queue successfully stored documents for the existing RAG worker."""
    queued = 0
    for item in results:
        if item.status != "imported" or not item.document_id:
            continue
        request = ingestion_request_factory(
            document_id=item.document_id,
            document_url=item.document_url or "",
            document_name=item.document_name or "Academic paper",
            document_type=item.document_type or "document",
        )
        background_tasks.add_task(ingestion_task, request, user_id)
        queued += 1
    return queued
