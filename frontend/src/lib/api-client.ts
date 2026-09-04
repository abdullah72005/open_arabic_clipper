export type RightsStatus =
  | "UNKNOWN"
  | "OWNED"
  | "LICENSED"
  | "PERMISSION"
  | "PUBLIC_DOMAIN"
  | "OTHER_ALLOWED";
export type PipelineStage = "INGEST" | "PROBE" | "READY_FOR_TRANSCRIPTION" | "FAILED";
export type JobStatus = "QUEUED" | "RUNNING" | "SUCCEEDED" | "FAILED" | "CANCELLED";

export interface Source {
  id: string;
  source_uri: string;
  original_filename: string | null;
  rights_status: RightsStatus;
  lifecycle_state: PipelineStage;
  created_at: string;
}

export interface Job {
  id: string;
  source_video_id: string;
  kind: string;
  status: JobStatus;
  retry_count: number;
  error_code: string | null;
  error_message: string | null;
}

export interface Health {
  status: "HEALTHY" | "DEGRADED" | "FAILED";
  checks: Array<{ name: string; status: string; detail?: string }>;
}

export interface StorageUsage {
  total_bytes: number;
  used_bytes: number;
  free_bytes: number;
}

type Fetcher = typeof fetch;

export class ApiError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

async function readResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as { detail?: string };
    throw new ApiError(response.status, body.detail ?? `Request failed (${response.status})`);
  }
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T);
}

export function createApiClient(baseUrl: string, fetcher: Fetcher = fetch) {
  const request = <T>(path: string, init?: RequestInit) =>
    fetcher(`${baseUrl.replace(/\/$/, "")}${path}`, init).then(readResponse<T>);

  return {
    listSources: () => request<Source[]>("/sources"),
    getSource: (id: string) => request<Source>(`/sources/${encodeURIComponent(id)}`),
    submitUrls: (urls: string[], rights_status: RightsStatus = "UNKNOWN") =>
      Promise.all(
        urls.map((url) =>
          request<Source>("/sources/url", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ url, rights_status })
          })
        )
      ),
    upload: (file: File, rights_status: RightsStatus = "UNKNOWN") => {
      const body = new FormData();
      body.append("file", file);
      body.append("rights_status", rights_status);
      return request<Source>("/sources/upload", { method: "POST", body });
    },
    deleteSource: (id: string) => request<void>(`/sources/${encodeURIComponent(id)}`, { method: "DELETE" }),
    listJobs: () => request<Job[]>("/jobs"),
    getJob: (id: string) => request<Job>(`/jobs/${encodeURIComponent(id)}`),
    cancelJob: (id: string) => request<Job>(`/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" }),
    health: () => request<Health>("/system/health"),
    storage: () => request<StorageUsage>("/system/storage")
  };
}

export const api = createApiClient(process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8300");
