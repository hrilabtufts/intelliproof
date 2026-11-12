import type { NextApiRequest, NextApiResponse } from "next";
import { proxyAutoExploreRequest } from "../../../lib/server/autoExploreBackendProxy";

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  return proxyAutoExploreRequest(req, res, "search");
}
