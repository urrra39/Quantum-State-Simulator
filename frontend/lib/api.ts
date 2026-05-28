import type {
  ApiError,
  SimulationRequest,
  SimulationResponse,
} from "@/lib/types";

// In the browser we go through Next.js rewrites at /api/tensorq/*
// (see next.config.mjs). This avoids CORS issues entirely.
const BASE = "/api/tensorq";

export class TensorQApiError extends Error {
  public readonly status: number;
  public readonly code: string;

  public constructor(status: number, code: string, message: string) {
    super(message);
    this.name = "TensorQApiError";
    this.status = status;
    this.code = code;
  }
}

const parseError = async (res: Response): Promise<TensorQApiError> => {
  let code = "http_error";
  let message = `HTTP ${res.status} ${res.statusText}`;
  try {
    const body = (await res.json()) as { detail?: ApiError | string };
    if (body && typeof body.detail === "object" && body.detail !== null) {
      code = body.detail.error ?? code;
      message = body.detail.message ?? message;
    } else if (typeof body?.detail === "string") {
      message = body.detail;
    }
  } catch {
    /* response body was not JSON */
  }
  return new TensorQApiError(res.status, code, message);
};

export const runSimulation = async (
  req: SimulationRequest,
  signal?: AbortSignal,
): Promise<SimulationResponse> => {
  const res = await fetch(`${BASE}/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as SimulationResponse;
};
