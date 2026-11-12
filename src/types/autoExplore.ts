export type AcademicDatabase = "arxiv" | "semantic_scholar" | "pubmed";

export interface NormalizedPaper {
  id: string;
  source: AcademicDatabase;
  title: string;
  abstract?: string | null;
  authors: string[];
  year?: number | null;
  publication_date?: string | null;
  doi?: string | null;
  url: string;
  pdf_url?: string | null;
  citation_count?: number | null;
  venue?: string | null;
  journal?: string | null;
  is_open_access?: boolean | null;
  fields_of_study?: string[];
  publication_types?: string[];
  external_ids: Record<string, string>;
}

export interface AutoExploreClaimMatch {
  claim_id: string;
  relation: "supports" | "challenges" | "contextualizes";
  rationale: string;
  confidence: number;
}

export interface AutoExplorePaperMatch {
  paper: NormalizedPaper;
  claim_matches: AutoExploreClaimMatch[];
}

export interface AutoExploreResponse {
  queries: string[];
  matches: AutoExplorePaperMatch[];
  errors: Partial<Record<AcademicDatabase, string>>;
  summary: string;
}

export interface AutoExploreImportItem {
  paper_id: string;
  status: "imported" | "duplicate" | "failed";
  storage_format?: "pdf" | "text" | null;
  fallback_reason?: string | null;
  document_id?: string | null;
  document_name?: string | null;
  document_url?: string | null;
  document_type?: string | null;
  indexing_status?: string | null;
  error?: string | null;
}

export interface AutoExploreImportResponse {
  results: AutoExploreImportItem[];
  summary: string;
}

export interface AutoExploreGraphSnapshot {
  graph_data: {
    nodes: Array<Record<string, unknown>>;
    edges: Array<Record<string, unknown>>;
    evidence: Array<Record<string, unknown>>;
    supportingDocuments: Array<Record<string, unknown>>;
  };
  key: string;
}

export async function importAutoExploreMatches(options: {
  graphId: string;
  accessToken: string;
  matches: AutoExplorePaperMatch[];
  refreshDocuments: () => Promise<unknown>;
  request?: typeof fetch;
}): Promise<AutoExploreImportResponse & { refreshError?: string }> {
  const request = options.request || fetch;
  const response = await request("/api/ai/auto-explore/import", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${options.accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      graph_id: options.graphId,
      selected_matches: options.matches,
    }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail =
      payload && typeof payload === "object"
        ? (payload as { detail?: string; error?: string }).detail ||
          (payload as { error?: string }).error
        : undefined;
    throw new Error(detail || "Could not add selected papers.");
  }

  const result = payload as Partial<AutoExploreImportResponse>;
  let refreshError: string | undefined;
  try {
    await options.refreshDocuments();
  } catch (error) {
    refreshError =
      error instanceof Error ? error.message : "Could not refresh graph documents.";
  }
  return {
    results: Array.isArray(result.results) ? result.results : [],
    summary: result.summary || "Paper import finished.",
    refreshError,
  };
}

type SnapshotNode = {
  id: string;
  data: { text: string; type?: string; evidenceIds?: string[] };
};

type SnapshotEdge = {
  id: string;
  source: string;
  target: string;
  data?: { edgeType?: string; confidence?: number };
};

type SnapshotEvidence = {
  id: string;
  title: string;
  supportingDocId: string;
  supportingDocName: string;
  excerpt: string;
};

type SnapshotDocument = {
  id: string;
  name: string;
  type: string;
  url: string;
  metadata?: unknown;
};

export function buildAutoExploreGraphSnapshot(input: {
  nodes: SnapshotNode[];
  edges: SnapshotEdge[];
  evidence: SnapshotEvidence[];
  documents: SnapshotDocument[];
}): AutoExploreGraphSnapshot {
  const graph_data = {
    nodes: input.nodes.map((node) => ({
      id: node.id,
      data: {
        text: node.data.text,
        type: node.data.type,
        evidenceIds: node.data.evidenceIds || [],
      },
    })),
    edges: input.edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      data: { relation: edge.data?.edgeType },
      weight: edge.data?.confidence,
    })),
    evidence: input.evidence.map((item) => ({
      id: item.id,
      title: item.title,
      supportingDocId: item.supportingDocId,
      supportingDocName: item.supportingDocName,
      excerpt: item.excerpt,
    })),
    supportingDocuments: input.documents.map((document) => ({
      id: document.id,
      name: document.name,
      type: document.type,
      url: document.url,
      metadata: document.metadata,
    })),
  };
  return { graph_data, key: JSON.stringify(graph_data) };
}
