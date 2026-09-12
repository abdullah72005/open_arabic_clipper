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
  | "CANDIDATE_ANALYSIS"
  | "READY_FOR_REFINEMENT"
  | "FAILED";
export type JobStatus = "QUEUED" | "RUNNING" | "SUCCEEDED" | "FAILED" | "CANCELLED";
export type ProviderAvailability = "AVAILABLE" | "UNAVAILABLE" | "MISCONFIGURED";
export type ReconstructionStatus =
  | "NOT_REQUIRED"
  | "APPLIED"
  | "UNCHANGED_HIGH_CONFIDENCE"
  | "LOW_CONFIDENCE_UNRESOLVED"
  | "PROVIDER_UNAVAILABLE"
  | "FAILED"
  | "MANUAL_OVERRIDE";

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
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
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
  reconstruction_status?: ReconstructionStatus;
  reconstruction_method?: string;
  routing_score?: number | null;
  routing_reasons?: string[];
  focus_spans?: Array<{
    word: string;
    start: number | null;
    end: number | null;
    probability: number | null;
  }>;
  avg_logprob?: number | null;
  no_speech_prob?: number | null;
}

export interface Transcript {
  source_video_id: string;
  language: string | null;
  detected_language_probability: number | null;
  whisper_model: string;
  transcription_options: Record<string, unknown>;
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
  reconstruction_status: ReconstructionStatus;
  segments: TranscriptSegment[];
  duration: number;
}

export interface QualityMetrics {
  audio_quality_score: number;
  transcript_quality_score: number;
  low_confidence_word_ratio: number;
  unresolved_segment_ratio: number;
  manual_review_required: boolean;
  conservative_source_floor: number;
}

export interface QualityResponse {
  reconstruction_status: ReconstructionStatus | null;
  quality: QualityMetrics | null;
}

export type CandidateDisposition =
  | "CANDIDATE"
  | "CANDIDATE_NEEDS_REFINEMENT"
  | "DO_NOT_CLIP"
  | "DO_NOT_CLIP_RECENTLY_REDUNDANT";

export interface Candidate {
  id: string;
  source_video_id: string;
  candidate_key: string;
  disposition: CandidateDisposition;
  start_time: number;
  end_time: number;
  transcript_excerpt: string;
  primary_content_type: string;
  clip_score: number;
  transcript_confidence: number;
  uncertainty_severity: number;
  refinement_reasons: string[];
  dialect_profile: string | null;
  code_switch_suspected: boolean;
}

export interface CandidateAnalysis {
  provider_status: string;
  semantic_provider_mode: string;
  cache_eligible: boolean;
  metrics: Record<string, unknown>;
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
    getQuality: (id: string) =>
      request<QualityResponse>(`/api/sources/${encodeURIComponent(id)}/quality`),
    getCandidateAnalysis: (id: string) =>
      request<CandidateAnalysis>(`/api/sources/${encodeURIComponent(id)}/candidate-analysis`),
    listCandidates: (id: string, options: { includeRejected?: boolean; limit?: number } = {}) =>
      request<Candidate[]>(
        `/api/sources/${encodeURIComponent(id)}/candidates?limit=${options.limit ?? 50}&include_rejected=${
          options.includeRejected ?? false
        }`
      ),
    retranscribeTranscript: (id: string, force = true) =>
      request<Job>(`/api/sources/${encodeURIComponent(id)}/retranscribe?force=${force}`, {
        method: "POST"
      }),
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
