import type { NextApiRequest, NextApiResponse } from "next";

const getBackendUrl = () => {
  // Match the backend routing used by the existing AI API routes. The public
  // frontend API URL can be a browser/LAN address that the Next server cannot
  // reach, so backend overrides for this server-side proxy use RAG_BACKEND_URL.
  const configuredUrl = process.env.RAG_BACKEND_URL?.trim();
  if (configuredUrl) return configuredUrl.replace(/\/$/, "");

  return process.env.NODE_ENV === "production"
    ? "https://intelliproofbackend.vercel.app"
    : "http://127.0.0.1:8000";
};

export async function proxyAutoExploreRequest(
  req: NextApiRequest,
  res: NextApiResponse,
  action: "search" | "import"
) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const response = await fetch(
      `${getBackendUrl()}/api/ai/auto-explore${action === "import" ? "/import" : ""}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(req.headers.authorization
            ? { Authorization: req.headers.authorization }
            : {}),
        },
        body: JSON.stringify(req.body),
        signal: AbortSignal.timeout(300000),
      }
    );
    const responseText = await response.text();
    let responseData: unknown;
    try {
      responseData = JSON.parse(responseText);
    } catch {
      responseData = { error: responseText || "Invalid backend response" };
    }
    return res.status(response.status).json(responseData);
  } catch (error) {
    console.error(`Auto-Explore ${action} proxy error:`, error);
    return res.status(502).json({
      error:
        error instanceof Error &&
        (error.name === "TimeoutError" || error.name === "AbortError")
          ? "Auto-Explore request timed out"
          : "Could not reach the Auto-Explore service",
    });
  }
}
