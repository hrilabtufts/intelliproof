"use client";

import { useEffect, useMemo, useState } from "react";
import type {
  AutoExploreImportItem,
  AutoExplorePaperMatch,
  AutoExploreResponse,
} from "../../types/autoExplore";

type AutoExploreResultsModalProps = {
  open: boolean;
  result: AutoExploreResponse | null;
  claimLabels: Record<string, string>;
  stale: boolean;
  isImporting: boolean;
  error: string | null;
  outcomes: AutoExploreImportItem[];
  onImport: (matches: AutoExplorePaperMatch[]) => void;
  onClose: () => void;
};

const sourceLabels: Record<string, string> = {
  arxiv: "arXiv",
  semantic_scholar: "Semantic Scholar",
  pubmed: "PubMed",
};

const safeExternalHref = (value?: string | null, httpsOnly = false) => {
  if (!value) return null;
  try {
    const parsed = new URL(value);
    if (httpsOnly ? parsed.protocol !== "https:" : !["http:", "https:"].includes(parsed.protocol)) {
      return null;
    }
    return parsed.href;
  } catch {
    return null;
  }
};

export default function AutoExploreResultsModal({
  open,
  result,
  claimLabels,
  stale,
  isImporting,
  error,
  outcomes,
  onImport,
  onClose,
}: AutoExploreResultsModalProps) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    setSelectedIds(new Set());
  }, [result]);

  useEffect(() => {
    if (!open) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !isImporting) onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [open, isImporting, onClose]);

  const outcomesByPaper = useMemo(
    () => new Map(outcomes.map((outcome) => [outcome.paper_id, outcome])),
    [outcomes]
  );

  if (!open || !result) return null;

  const selectableMatches = result.matches.filter((match) => {
    const status = outcomesByPaper.get(match.paper.id)?.status;
    return status !== "imported" && status !== "duplicate";
  });
  const selectedMatches = selectableMatches.filter((match) =>
    selectedIds.has(match.paper.id)
  );
  const allSelected =
    selectableMatches.length > 0 &&
    selectableMatches.every((match) => selectedIds.has(match.paper.id));

  const togglePaper = (paperId: string) => {
    setSelectedIds((current) => {
      const next = new Set(current);
      if (next.has(paperId)) next.delete(paperId);
      else next.add(paperId);
      return next;
    });
  };

  return (
    <div
      className="fixed inset-0 z-[80] flex items-center justify-center bg-black/40 p-4"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !isImporting) onClose();
      }}
    >
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="auto-explore-results-title"
        className="flex max-h-[90vh] w-full max-w-4xl flex-col overflow-hidden rounded-xl bg-white shadow-2xl"
      >
        <header className="flex items-start justify-between gap-4 border-b border-gray-200 p-5">
          <div>
            <h2
              id="auto-explore-results-title"
              className="text-lg font-semibold text-gray-900"
            >
              Auto-Explore results
            </h2>
            <p className="mt-1 text-sm text-gray-600">{result.summary}</p>
            <p className="mt-1 text-xs text-gray-500">
              Relevance confidence is not evidence strength.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={isImporting}
            className="rounded px-2 py-1 text-sm text-gray-600 hover:bg-gray-100"
            aria-label="Close Auto-Explore results"
          >
            Close
          </button>
        </header>

        <div className="space-y-3 overflow-y-auto p-5">
          {result.queries.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-gray-800">
                Searches used
              </h3>
              <ul className="mt-1 list-inside list-disc text-sm text-gray-600">
                {result.queries.map((query, index) => (
                  <li key={`${query}-${index}`}>{query}</li>
                ))}
              </ul>
            </div>
          )}

          {Object.entries(result.errors).length > 0 && (
            <div
              role="status"
              className="rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900"
            >
              {Object.entries(result.errors).map(([source, message]) => (
                <p key={source}>
                  <strong>{sourceLabels[source] || source}:</strong> {message}
                </p>
              ))}
            </div>
          )}

          {stale && (
            <div
              role="alert"
              className="rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900"
            >
              The graph or its documents changed during this search. Run
              Auto-Explore again before importing these results.
            </div>
          )}

          {error && (
            <div role="alert" className="rounded-md bg-red-50 p-3 text-sm text-red-700">
              {error}
            </div>
          )}

          <div className="flex items-center justify-between gap-3 border-b border-gray-200 pb-2">
            <h3 className="text-sm font-semibold text-gray-900">
              Matched papers ({result.matches.length})
            </h3>
            <button
              type="button"
              disabled={!selectableMatches.length || isImporting || stale}
              onClick={() =>
                setSelectedIds(
                  allSelected
                    ? new Set()
                    : new Set(selectableMatches.map((match) => match.paper.id))
                )
              }
              className="text-sm font-medium text-[#232F3E] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {allSelected ? "Clear selection" : "Select all"}
            </button>
          </div>

          {result.matches.length === 0 ? (
            <p className="rounded-md bg-gray-50 p-4 text-sm text-gray-600">
              No new papers matched the graph claims.
            </p>
          ) : (
            <ul className="space-y-3">
              {result.matches.map((match, index) => {
                const { paper } = match;
                const paperHref = safeExternalHref(paper.url);
                const pdfHref = safeExternalHref(paper.pdf_url, true);
                const outcome = outcomesByPaper.get(paper.id);
                const disabled =
                  isImporting ||
                  stale ||
                  outcome?.status === "imported" ||
                  outcome?.status === "duplicate";
                return (
                  <li
                    key={paper.id}
                    className="rounded-lg border border-gray-200 p-4"
                  >
                    <div className="flex items-start gap-3">
                      <input
                        id={`auto-explore-paper-${index}`}
                        type="checkbox"
                        aria-label={`Select ${paper.title}`}
                        checked={selectedIds.has(paper.id)}
                        disabled={disabled}
                        onChange={() => {
                          if (!disabled) togglePaper(paper.id);
                        }}
                        className="mt-1 h-4 w-4 accent-[#232F3E]"
                      />
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-start justify-between gap-2">
                          <h4 className="font-semibold text-gray-900">
                            {paperHref ? (
                              <a
                                href={paperHref}
                                target="_blank"
                                rel="noreferrer"
                                className="underline decoration-gray-400 underline-offset-2 hover:text-blue-700"
                              >
                                {paper.title}
                              </a>
                            ) : (
                              paper.title
                            )}
                          </h4>
                          <span className="rounded bg-gray-100 px-2 py-1 text-xs text-gray-700">
                            {sourceLabels[paper.source] || paper.source}
                          </span>
                        </div>
                        <p className="mt-1 text-xs text-gray-600">
                          {[paper.authors.slice(0, 5).join(", "), paper.year, paper.venue || paper.journal]
                            .filter(Boolean)
                            .join(" · ")}
                        </p>
                        {paper.abstract && (
                          <p className="mt-2 whitespace-pre-line text-sm text-gray-700">
                            {paper.abstract}
                          </p>
                        )}
                        {!paper.abstract && (
                          <p className="mt-2 text-sm italic text-gray-500">
                            No abstract is available from this source.
                          </p>
                        )}
                        {pdfHref && paper.is_open_access !== false && (
                          <a
                            href={pdfHref}
                            target="_blank"
                            rel="noreferrer"
                            className="mt-2 inline-block text-xs font-medium text-blue-700 underline"
                          >
                            PDF available
                          </a>
                        )}

                        <div className="mt-3 space-y-2">
                          {match.claim_matches.map((claimMatch) => (
                            <div
                              key={`${paper.id}-${claimMatch.claim_id}`}
                              className="rounded bg-gray-50 p-2 text-sm"
                            >
                              <div className="flex flex-wrap justify-between gap-2 text-xs font-medium text-gray-700">
                                <span>
                                  {claimLabels[claimMatch.claim_id] ||
                                    `Claim ${claimMatch.claim_id}`}
                                </span>
                                <span>
                                  {claimMatch.relation} · {Math.round(claimMatch.confidence * 100)}% relevance
                                </span>
                              </div>
                              <p className="mt-1 text-gray-600">
                                {claimMatch.rationale}
                              </p>
                            </div>
                          ))}
                        </div>

                        {outcome && (
                          <p
                            role={outcome.status === "failed" ? "alert" : "status"}
                            className={`mt-3 text-sm ${
                              outcome.status === "failed"
                                ? "text-red-700"
                                : "text-green-700"
                            }`}
                          >
                            {outcome.status === "imported" &&
                              `Added as ${outcome.storage_format === "pdf" ? "PDF" : "text document"}; indexing ${outcome.indexing_status || "pending"}.`}
                            {outcome.status === "duplicate" && "Already in this graph’s Documents."}
                            {outcome.status === "failed" && outcome.error}
                            {outcome.fallback_reason && (
                              <span className="block text-xs text-gray-600">
                                {outcome.fallback_reason}
                              </span>
                            )}
                          </p>
                        )}
                      </div>
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </div>

        <footer className="flex items-center justify-between gap-3 border-t border-gray-200 p-4">
          <p className="text-sm text-gray-600">
            {outcomes.length > 0
              ? `${outcomes.filter((item) => item.status === "imported").length} imported · ${outcomes.filter((item) => item.status === "duplicate").length} already present · ${outcomes.filter((item) => item.status === "failed").length} failed`
              : `${selectedMatches.length} selected`}
          </p>
          <button
            type="button"
            disabled={!selectedMatches.length || isImporting || stale}
            onClick={() => onImport(selectedMatches)}
            className="rounded-md bg-[#232F3E] px-4 py-2 text-sm font-medium text-white hover:bg-[#1A2330] disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isImporting ? "Adding documents…" : `Add selected (${selectedMatches.length})`}
          </button>
        </footer>
      </section>
    </div>
  );
}
