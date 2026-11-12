import { describe, expect, it, vi } from "vitest";
import {
  buildAutoExploreGraphSnapshot,
  importAutoExploreMatches,
} from "./autoExplore";

const graphInput = {
  nodes: [{ id: "n1", data: { text: "Claim", type: "factual", evidenceIds: [] } }],
  edges: [{ id: "e1", source: "n1", target: "n2", data: { edgeType: "supporting", confidence: 0.7 } }],
  evidence: [
    { id: "ev1", title: "Passage", supportingDocId: "d1", supportingDocName: "source.pdf", excerpt: "Exact quote" },
  ],
  documents: [
    { id: "d1", name: "source.pdf", type: "document", url: "https://storage/doc", metadata: { doi: "10.1234/source" } },
  ],
};

describe("Auto-Explore graph and import workflow helpers", () => {
  it("includes claim, connection, evidence, and document changes in the stale key", () => {
    const initial = buildAutoExploreGraphSnapshot(graphInput);
    const updated = buildAutoExploreGraphSnapshot({
      ...graphInput,
      edges: [{ ...graphInput.edges[0], data: { edgeType: "attacking", confidence: -0.4 } }],
    });
    const documentChange = buildAutoExploreGraphSnapshot({
      ...graphInput,
      documents: [{ ...graphInput.documents[0], name: "other.pdf" }],
    });

    expect(initial.key).not.toBe(updated.key);
    expect(initial.key).not.toBe(documentChange.key);
    expect(initial.graph_data.supportingDocuments[0].metadata).toEqual({ doi: "10.1234/source" });
  });

  it("refreshes the graph document list after successful import", async () => {
    const refreshDocuments = vi.fn().mockResolvedValue(undefined);
    const request = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        results: [{ paper_id: "arxiv:1", status: "imported", storage_format: "pdf" }],
        summary: "Imported one paper.",
      }),
    });

    const outcome = await importAutoExploreMatches({
      graphId: "graph-1",
      accessToken: "session-token",
      matches: [],
      refreshDocuments,
      request: request as typeof fetch,
    });

    expect(request).toHaveBeenCalledWith(
      "/api/ai/auto-explore/import",
      expect.objectContaining({ method: "POST" })
    );
    expect(refreshDocuments).toHaveBeenCalledOnce();
    expect(outcome.results[0].status).toBe("imported");
  });

  it("does not refresh the list when the import request fails", async () => {
    const refreshDocuments = vi.fn();
    await expect(
      importAutoExploreMatches({
        graphId: "graph-1",
        accessToken: "session-token",
        matches: [],
        refreshDocuments,
        request: vi.fn().mockResolvedValue({
          ok: false,
          json: async () => ({ detail: "Graph access denied" }),
        }) as typeof fetch,
      })
    ).rejects.toThrow("Graph access denied");
    expect(refreshDocuments).not.toHaveBeenCalled();
  });

  it("preserves import outcomes when refreshing graph documents fails", async () => {
    const outcome = await importAutoExploreMatches({
      graphId: "graph-1",
      accessToken: "session-token",
      matches: [],
      refreshDocuments: async () => {
        throw new Error("Document list unavailable");
      },
      request: vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          results: [{ paper_id: "arxiv:1", status: "imported", storage_format: "pdf" }],
          summary: "Imported one paper.",
        }),
      }) as typeof fetch,
    });

    expect(outcome.results[0].status).toBe("imported");
    expect(outcome.refreshError).toBe("Document list unavailable");
  });
});
