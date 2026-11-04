"""Shared, provider-neutral models for academic search results."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


AcademicSource = Literal["arxiv", "semantic_scholar", "pubmed"]


class NormalizedPaper(BaseModel):
    """A paper returned by one of IntelliProof's academic search providers.

    ``id`` is a stable provider-scoped key (``arxiv:<id>``, ``s2:<paperId>`` or
    ``pmid:<id>``). Keep provider identifiers in ``external_ids`` as well so
    callers can link records across providers without parsing the display ID.
    Abstracts are optional because many indexed records do not include one.
    """

    id: str = Field(description="Stable, provider-scoped identifier")
    source: AcademicSource
    title: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=1000, le=9999)
    publication_date: str | None = None
    doi: str | None = None
    url: str = Field(description="Canonical landing page for this paper")
    pdf_url: str | None = None
    citation_count: int | None = Field(default=None, ge=0)
    venue: str | None = None
    journal: str | None = None
    is_open_access: bool | None = None
    fields_of_study: list[str] = Field(default_factory=list)
    publication_types: list[str] = Field(default_factory=list)
    external_ids: dict[str, str] = Field(default_factory=dict)


class AcademicSearchResponse(BaseModel):
    """Combined search output, including per-provider failures.

    A failure in one provider does not prevent returning useful results from
    the other providers. Consumers can display ``errors`` as source-specific
    status while still using ``papers`` in chat or graph workflows.
    """

    query: str
    papers: list[NormalizedPaper] = Field(default_factory=list)
    errors: dict[AcademicSource, str] = Field(default_factory=dict)
