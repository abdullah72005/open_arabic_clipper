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

export type RefinementPriority = "CANDIDATE" | "FINAL_CLIP";

export interface CandidateRefinement {
  id: string;
  source_video_id: string;
  clip_candidate_id: string;
  priority: RefinementPriority;
  status: string;
  quality_level: string;
  coarse_start: number;
  coarse_end: number;
  context_start: number;
  context_end: number;
  refined_start: number | null;
  refined_end: number | null;
  automatic_transcript: string;
  manual_transcript: string | null;
  final_transcript: string;
  word_timestamps: Array<Record<string, unknown>>;
  confidence: number;
  dialect_profile: string | null;
  dialect_confidence: number;
  code_switch_evidence: Record<string, unknown>;
  transcript_evidence: Array<Record<string, unknown>>;
  entity_evidence: Array<Record<string, unknown>>;
  unresolved_spans: Array<Record<string, unknown>>;
  provider_evidence: Record<string, unknown>;
  routing_evidence: Record<string, unknown>;
  input_fingerprint: string;
  output_fingerprint: string;
  cache_eligible: boolean;
  metrics: Record<string, unknown>;
  processing_duration: number | null;
  created_at: string;
  updated_at: string;
}

export interface RefinementQueueResponse {
  refinement_id: string;
  job_id: string | null;
  status: string;
  queued: boolean;
  cached: boolean;
  active: boolean;
}

export interface TransformationQueueResponse {
  analysis_id: string;
  job_id: string | null;
  status: string;
  queued: boolean;
  cached: boolean;
  active: boolean;
}

export interface TransformationAnalysis {
  id: string;
  execution_status: string;
  eligibility_outcome: string | null;
  eligibility_reasons: string[];
  cache_eligible: boolean;
  recommended_strategies: Array<Record<string, unknown>>;
  rejected_strategies: Array<Record<string, unknown>>;
}

export interface PlanningQueueResponse {
  plan_set_id: string;
  job_id: string | null;
  status: string;
  queued: boolean;
  cached: boolean;
  active: boolean;
}

export interface TransformationPlan {
  id: string;
  plan_key: string;
  status: string;
  strategy_type: string;
  intensity: string;
  planner_confidence: number;
}

export interface TransformationPlanSet {
  id: string;
  execution_status: string;
  planning_outcome: string | null;
  outcome_reasons: string[];
  cache_eligible: boolean;
  plans: TransformationPlan[];
}

export interface GovernanceQueueResponse {
  governance_set_id: string;
  job_id: string | null;
  status: string;
  queued: boolean;
  cached: boolean;
  active: boolean;
}

export interface GovernanceSet {
  id: string;
  execution_status: string;
  semantic_outcome: string | null;
  outcome_reasons: string[];
  results: Array<Record<string, unknown>>;
}

export interface TransformationSelection {
  id: string;
  selected_plan_id: string | null;
  status: string;
  is_current: boolean;
  effective: boolean;
  selected_with_caution: boolean;
  selection_reason_codes: string[];
  live_freshness: string;
}

export interface RenderContract {
  id: string;
  status: string;
  is_current: boolean;
  effective: boolean;
  contract_ready: boolean;
  compatibility_outcome: string | null;
  reason_codes: string[];
  live_freshness: string;
}

export interface VisualCompositionQueueResponse {
  plan_id: string;
  job_id: string | null;
  status: string;
  queued: boolean;
  cached: boolean;
  active: boolean;
}

export interface VisualComposition {
  id: string;
  status: string;
  execution_status: string;
  plan_ready: boolean;
  is_current: boolean;
  effective: boolean;
  cache_eligible: boolean;
  reason_codes: string[];
  live_freshness: string;
}

export interface RenderExecutionQueueResponse {
  render_execution_id: string;
  job_id: string | null;
  status: string;
  queued: boolean;
  cached: boolean;
  active: boolean;
  artifact_purpose: string;
  delivery_profile_key: string;
}

export interface RenderExecution {
  id: string;
  lifecycle: string;
  qc_status: string | null;
  is_current: boolean;
  cache_eligible: boolean;
  effective: boolean;
  publication_ready: boolean;
  reason_codes: string[];
  error_code: string | null;
  live_freshness: string;
}

export interface RenderArtifact {
  id: string;
  source_video_id: string;
  clip_candidate_id: string;
  artifact_purpose: string;
  lifecycle: string;
  qc_status: string | null;
  cache_eligible: boolean;
  is_current: boolean;
  created_at: string;
}

export interface MaterializationReadiness {
  review_ready: boolean;
  reasons: string[];
  freshness?: string;
  is_effective?: boolean;
  outcome?: string | null;
  publication_ready?: boolean;
}

export interface Materialization {
  id: string;
  source_video_id: string;
  clip_candidate_id: string;
  profile_key: string;
  priority: string;
  lifecycle: string;
  semantic_outcome: string | null;
  human_review_required: boolean;
  publication_ready: boolean;
  is_current: boolean;
  repair_count: number;
  reason_codes: string[];
  input_fingerprint: string;
  output_fingerprint: string;
  policy_version: string;
  schema_version: string;
  render_execution_id: string | null;
  selection_id: string | null;
  render_contract_id: string | null;
  is_effective: boolean;
  live_freshness: string;
  readiness: MaterializationReadiness;
  component_statuses: Record<string, unknown>;
  findings: Array<Record<string, unknown>>;
  provider_attempts: Record<string, unknown>;
  artifact_available: boolean;
}

export interface MaterializationQueueResponse {
  job_id: string | null;
  status: string;
  queued: boolean;
  cached: boolean;
  active: boolean;
  materialization_id: string | null;
  render_execution_id: string | null;
  phase: string | null;
  lifecycle: string | null;
  semantic_outcome: string | null;
  human_review_required: boolean;
  publication_ready: boolean;
  review_ready: boolean;
  reasons: string[];
  skipped_reason: string | null;
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
    listCandidateRefinements: (candidateId: string) =>
      request<CandidateRefinement[]>(`/api/candidates/${encodeURIComponent(candidateId)}/refinements`),
    queueCandidateRefinement: (candidateId: string, priority: RefinementPriority) =>
      request<RefinementQueueResponse>(
        `/api/candidates/${encodeURIComponent(candidateId)}/refinements?priority=${priority}`,
        { method: "POST" }
      ),
    submitManualCandidateTranscript: (
      refinementId: string,
      text: string,
      resolutions: Record<string, string>
    ) =>
      request<CandidateRefinement>(`/api/refinements/${encodeURIComponent(refinementId)}/manual`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ text, resolutions })
      }),
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
    queueTransformationAnalysis: (candidateId: string, force = false) =>
      request<TransformationQueueResponse>(
        `/api/candidates/${encodeURIComponent(candidateId)}/transformation-analyses?force=${force}`,
        { method: "POST" }
      ),
    getCandidateTransformationAnalysis: (candidateId: string) =>
      request<TransformationAnalysis>(
        `/api/candidates/${encodeURIComponent(candidateId)}/transformation-analysis`
      ),
    queueTransformationPlans: (candidateId: string, force = false) =>
      request<PlanningQueueResponse>(
        `/api/candidates/${encodeURIComponent(candidateId)}/transformation-plans?force=${force}`,
        { method: "POST" }
      ),
    getCandidateTransformationPlans: (candidateId: string) =>
      request<TransformationPlanSet>(
        `/api/candidates/${encodeURIComponent(candidateId)}/transformation-plans`
      ),
    queueTransformationGovernance: (candidateId: string, force = false) =>
      request<GovernanceQueueResponse>(
        `/api/candidates/${encodeURIComponent(candidateId)}/transformation-governance?force=${force}`,
        { method: "POST" }
      ),
    getCandidateTransformationGovernance: (candidateId: string) =>
      request<GovernanceSet>(
        `/api/candidates/${encodeURIComponent(candidateId)}/transformation-governance`
      ),
    createTransformationSelection: (candidateId: string) =>
      request<TransformationSelection>(
        `/api/candidates/${encodeURIComponent(candidateId)}/transformation-selection`,
        { method: "POST" }
      ),
    createRenderContract: (candidateId: string) =>
      request<RenderContract>(
        `/api/candidates/${encodeURIComponent(candidateId)}/render-contract`,
        { method: "POST" }
      ),
    queueVisualComposition: (candidateId: string, force = false) =>
      request<VisualCompositionQueueResponse>(
        `/api/candidates/${encodeURIComponent(candidateId)}/visual-composition?force=${force}`,
        { method: "POST" }
      ),
    getCandidateVisualComposition: (candidateId: string) =>
      request<VisualComposition>(
        `/api/candidates/${encodeURIComponent(candidateId)}/visual-composition`
      ),
    queueRenderExecution: (candidateId: string, force = false) =>
      request<RenderExecutionQueueResponse>(
        `/api/candidates/${encodeURIComponent(candidateId)}/render-execution?force=${force}`,
        { method: "POST" }
      ),
    getCandidateRenderExecution: (candidateId: string) =>
      request<RenderExecution>(
        `/api/candidates/${encodeURIComponent(candidateId)}/render-execution`
      ),
    renderExecutionArtifactUrl: (renderExecutionId: string) =>
      `${baseUrl.replace(/\/$/, "")}/api/render-executions/${encodeURIComponent(renderExecutionId)}/artifact`,
    listRenderArtifacts: (limit = 50) =>
      request<RenderArtifact[]>(`/api/render-executions?limit=${limit}`),
    queueMaterialization: (
      candidateId: string,
      options: {
        priority?: string;
        profileKey?: string;
        supportingRequested?: string;
        force?: boolean;
      } = {}
    ) => {
      const query = new URLSearchParams({
        priority: options.priority ?? "CANDIDATE",
        supporting_requested: options.supportingRequested ?? "OFF",
        force: String(options.force ?? false)
      });
      if (options.profileKey) query.set("profile_key", options.profileKey);
      return request<MaterializationQueueResponse>(
        `/api/candidates/${encodeURIComponent(candidateId)}/materialization?${query.toString()}`,
        { method: "POST" }
      );
    },
    getCandidateMaterialization: (candidateId: string, profileKey?: string) =>
      request<Materialization>(
        `/api/candidates/${encodeURIComponent(candidateId)}/materialization${
          profileKey ? `?profile_key=${encodeURIComponent(profileKey)}` : ""
        }`
      ),
    getMaterialization: (materializationId: string) =>
      request<Materialization>(`/api/materializations/${encodeURIComponent(materializationId)}`),
    cancelMaterialization: (materializationId: string) =>
      request<Materialization>(
        `/api/materializations/${encodeURIComponent(materializationId)}/cancel`,
        { method: "POST" }
      ),
    resumeMaterialization: (materializationId: string) =>
      request<MaterializationQueueResponse>(
        `/api/materializations/${encodeURIComponent(materializationId)}/resume`,
        { method: "POST" }
      ),
    materializationArtifactUrl: (materializationId: string) =>
      `${baseUrl.replace(/\/$/, "")}/api/materializations/${encodeURIComponent(materializationId)}/artifact`,
    listJobs: () => request<Job[]>("/jobs"),
    getJob: (id: string) => request<Job>(`/jobs/${encodeURIComponent(id)}`),
    cancelJob: (id: string) => request<Job>(`/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" }),
    health: () => request<Health>("/system/health"),
    storage: () => request<StorageUsage>("/system/storage")
  };
}

export const api = createApiClient(process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8300");
