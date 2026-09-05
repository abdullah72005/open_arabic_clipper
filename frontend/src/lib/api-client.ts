export type RightsStatus =
  | "UNKNOWN"
  | "OWNED"
  | "LICENSED"
  | "PERMISSION"
  | "PUBLIC_DOMAIN"
  | "OTHER_ALLOWED"
  | "THIRD_PARTY_UNKNOWN"
  | "THIRD_PARTY_REUSE";
export type PipelineStage =
  | "INGEST"
  | "PROBE"
  | "AUDIO_EXTRACTION"
  | "TRANSCRIPTION"
  | "TRANSCRIPT_NORMALIZATION"
  | "CONTEXTUAL_RECONSTRUCTION"
  | "AUDIO_ANALYSIS"
  | "READY_FOR_TRANSCRIPTION"
  | "READY_FOR_ANALYSIS"
  | "FAILED";
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

export interface TranscriptSegment {
  start: number;
  end: number;
  text: string;
  normalized_text?: string;
  raw_text?: string;
  corrected_text?: string;
  final_text?: string;
  operator_text?: string | null;
  correction_applied?: boolean;
  correction_confidence?: number;
  correction_method?: string;
  correction_version?: string;
  contextual_reconstructed_text?: string;
  reconstruction_candidate_text?: string | null;
  reconstruction_applied?: boolean;
  reconstruction_confidence?: number;
  reconstruction_confidence_level?: "HIGH" | "MEDIUM" | "LOW";
  reconstruction_quality_flags?: string[];
  avg_logprob?: number | null;
  no_speech_prob?: number | null;
}

export interface Transcript {
  source_video_id: string;
  language: string | null;
  detected_language_probability: number | null;
  raw_text: string;
  normalized_text: string;
  corrected_text?: string;
  final_text?: string;
  raw_transcript_confidence?: number;
  correction_confidence?: number;
  corrected_segment_ratio?: number;
  uncertain_segment_ratio?: number;
  correction_method?: string;
  correction_version?: string;
  contextual_reconstructed_text?: string;
  reconstruction_fingerprint?: string;
  reconstruction_confidence?: number;
  reconstructed_segment_ratio?: number;
  reconstruction_method?: string;
  reconstruction_version?: string;
  reconstruction_processing_duration?: number | null;
  reconstruction_metadata?: Record<string, unknown>;
  segments: TranscriptSegment[];
  duration: number;
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
    sourceMediaUrl: (id: string) =>
      `${baseUrl.replace(/\/$/, "")}/api/sources/${encodeURIComponent(id)}/media`,
    getTranscript: (id: string) =>
      request<Transcript>(`/api/sources/${encodeURIComponent(id)}/transcript`),
    reconstructTranscript: (id: string, force = false) =>
      request<Job>(`/api/sources/${encodeURIComponent(id)}/reconstruct?force=${force}`, {
        method: "POST"
      }),
    overrideTranscriptSegment: (id: string, segmentIndex: number, text: string) =>
      request<TranscriptSegment>(
        `/api/sources/${encodeURIComponent(id)}/transcript/segments/${segmentIndex}/override`,
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ text })
        }
      ),
    clearTranscriptSegmentOverride: (id: string, segmentIndex: number) =>
      request<TranscriptSegment>(
        `/api/sources/${encodeURIComponent(id)}/transcript/segments/${segmentIndex}/override`,
        { method: "DELETE" }
      ),
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
    retrySource: (id: string) => request<Job>(`/sources/${encodeURIComponent(id)}/retry`, { method: "POST" }),
    listJobs: () => request<Job[]>("/jobs"),
    getJob: (id: string) => request<Job>(`/jobs/${encodeURIComponent(id)}`),
    cancelJob: (id: string) => request<Job>(`/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" }),
    health: () => request<Health>("/system/health"),
    storage: () => request<StorageUsage>("/system/storage")
  };
}

export const api = createApiClient(process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8300");
