import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import AutoExploreResultsModal from "./AutoExploreResultsModal";
import type {
  AutoExplorePaperMatch,
  AutoExploreResponse,
} from "../../types/autoExplore";

const match: AutoExplorePaperMatch = {
  paper: {
    id: "arxiv:123",
    source: "arxiv",
    title: "A paper about this claim",
    abstract: "A useful abstract.",
    authors: ["A. Author"],
    year: 2024,
    url: "https://arxiv.org/abs/123",
    pdf_url: "https://arxiv.org/pdf/123",
    is_open_access: true,
    external_ids: { arxiv: "123" },
  },
  claim_matches: [
    {
      claim_id: "claim-1",
      relation: "supports",
      rationale: "It examines the claim directly.",
      confidence: 0.8,
    },
  ],
};

const result: AutoExploreResponse = {
  queries: ["claim evidence"],
  matches: [match],
  errors: { pubmed: "Provider unavailable" },
  summary: "Found one relevant paper.",
};

const renderModal = (overrides = {}) =>
  render(
    <AutoExploreResultsModal
      open
      result={result}
      claimLabels={{ "claim-1": "The claim text" }}
      stale={false}
      isImporting={false}
      error={null}
      outcomes={[]}
      onImport={vi.fn()}
      onClose={vi.fn()}
      {...overrides}
    />
  );

describe("AutoExploreResultsModal", () => {
  it("starts with no papers selected and imports only selected matches", () => {
    const onImport = vi.fn();
    renderModal({ onImport });

    expect(screen.getByText("claim evidence")).toBeInTheDocument();
    expect(screen.getByText("The claim text")).toBeInTheDocument();
    expect(screen.getByText("It examines the claim directly.")).toBeInTheDocument();
    expect(screen.getByLabelText("Select A paper about this claim")).not.toBeChecked();
    expect(screen.getByRole("button", { name: "Add selected (0)" })).toBeDisabled();

    fireEvent.click(screen.getByLabelText("Select A paper about this claim"));
    fireEvent.click(screen.getByRole("button", { name: "Add selected (1)" }));
    expect(onImport).toHaveBeenCalledWith([match]);
  });

  it("blocks import for stale graph state and explains why", () => {
    renderModal({ stale: true });
    expect(screen.getByRole("alert")).toHaveTextContent("Run Auto-Explore again");
    expect(screen.getByLabelText("Select A paper about this claim")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Add selected (0)" })).toBeDisabled();
  });

  it("shows per-paper import and indexing results", () => {
    renderModal({
      outcomes: [
        {
          paper_id: "arxiv:123",
          status: "imported",
          storage_format: "text",
          indexing_status: "pending",
          fallback_reason: "No provider PDF was accessible",
        },
      ],
    });
    expect(screen.getByText(/Added as text document/).closest('[role="status"]')).toHaveTextContent("Added as text document");
    expect(screen.getByText("No provider PDF was accessible")).toBeInTheDocument();
    expect(screen.getByLabelText("Select A paper about this claim")).toBeDisabled();
  });
});
