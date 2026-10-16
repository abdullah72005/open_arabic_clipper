import { describe, expect, it } from "vitest";

import type {
  Candidate,
  CandidateRefinement,
  Job,
  Materialization,
  MaterializationQueueResponse,
  RightsStatus,
  Source
} from "@/lib/api-client";

import {
  runProduceChain,
  type ProduceApi,
  type ProduceCallbacks,
  type ProduceRun
} from "./produce-runner";

const source: Source = {
  id: "src-1",
  source_uri: "https://example.org/v",
  original_filename: null,
  rights_status: "UNKNOWN",
  lifecycle_state: "READY_FOR_ANALYSIS",
  created_at: "2026-10-01T00:00:00Z"
};

const job: Job = {
  id: "job-candidate-analysis",
  source_video_id: "src-1",
  kind: "CANDIDATE_ANALYSIS",
  status: "SUCCEEDED",
  retry_count: 0,
  error_code: null,
  error_message: null,
  created_at: "2026-10-01T00:00:00Z",
  started_at: "2026-10-01T00:00:01Z",
  completed_at: "2026-10-01T00:00:02Z"
};

const candidate: Candidate = {
  id: "cand-1",
  source_video_id: "src-1",
  candidate_key: "key-1",
  disposition: "CANDIDATE",
  start_time: 0,
  end_time: 30,
  transcript_excerpt: "text",
  primary_content_type: "STORY",
  clip_score: 0.7,
  transcript_confidence: 0.9,
  uncertainty_severity: 0.1,
  refinement_reasons: [],
  dialect_profile: "EGYPTIAN",
  code_switch_suspected: false
};

const refinement: CandidateRefinement = {
  id: "ref-1",
  source_video_id: "src-1",
  clip_candidate_id: "cand-1",
  priority: "FINAL_CLIP",
  status: "FINAL_TRANSCRIPT_READY",
  quality_level: "FINAL_CLIP",
  coarse_start: 0,
  coarse_end: 30,
  context_start: 0,
  context_end: 30,
  refined_start: 0,
  refined_end: 30,
  automatic_transcript: "text",
  manual_transcript: null,
  final_transcript: "text",
  word_timestamps: [],
  confidence: 0.9,
  dialect_profile: "EGYPTIAN",
  dialect_confidence: 0.9,
  code_switch_evidence: {},
  transcript_evidence: [],
  entity_evidence: [],
  unresolved_spans: [],
  provider_evidence: {},
  routing_evidence: {},
  input_fingerprint: "in",
  output_fingerprint: "out",
  cache_eligible: true,
  metrics: {},
  processing_duration: 1,
  created_at: "2026-10-01T00:00:00Z",
  updated_at: "2026-10-01T00:00:00Z"
};

function materialization(overrides: Partial<Materialization>): Materialization {
  return {
    id: "mat-1",
    source_video_id: "src-1",
    clip_candidate_id: "cand-1",
    profile_key: "default",
    priority: "CANDIDATE",
    lifecycle: "AWAITING_RENDER",
    semantic_outcome: null,
    human_review_required: false,
    publication_ready: false,
    is_current: true,
    repair_count: 0,
    reason_codes: [],
    input_fingerprint: "in",
    output_fingerprint: "out",
    policy_version: "stage6-v1",
    schema_version: "stage6-schema-v1",
    render_execution_id: null,
    selection_id: "sel-1",
    render_contract_id: "con-1",
    is_effective: false,
    live_freshness: "CURRENT",
    readiness: { review_ready: false, reasons: [], publication_ready: false },
    component_statuses: {},
    findings: [],
    provider_attempts: {},
    artifact_available: false,
    ...overrides
  };
}

class FakeApi implements ProduceApi {
  readonly materializations: Materialization[];
  private index = 0;
  readonly getMaterializationCalls: string[] = [];
  readonly cancelCalls: string[] = [];
  readonly resumeCalls: string[] = [];
  private cancelFailures = 0;
  private cancelAttempts = 0;
  deferSubmission = false;
  submissionStarted!: () => void;
  readonly submissionReady = new Promise<void>((resolve) => {
    this.submissionStarted = resolve;
  });
  private resolveSubmission: ((response: MaterializationQueueResponse) => void) | null = null;
  private materializationId = "mat-1";

  constructor(materializations: Materialization[]) {
    this.materializations = materializations;
  }

  private next(): Materialization {
    const current = this.materializations[Math.min(this.index, this.materializations.length - 1)];
    this.index += 1;
    return current;
  }

  submitUrls(): Promise<Source[]> {
    return Promise.resolve([source]);
  }
  listJobs(): Promise<Job[]> {
    return Promise.resolve([job]);
  }
  getJob(): Promise<Job> {
    return Promise.resolve(job);
  }
  getCandidateAnalysis(): Promise<unknown> {
    return Promise.resolve({});
  }
  listCandidates(): Promise<Candidate[]> {
    return Promise.resolve([candidate]);
  }
  queueCandidateRefinement(): Promise<{ job_id: string | null; status: string; cached: boolean }> {
    return Promise.resolve({ job_id: null, status: "SUCCEEDED", cached: true });
  }
  listCandidateRefinements(): Promise<CandidateRefinement[]> {
    return Promise.resolve([refinement]);
  }
  queueTransformationAnalysis(): Promise<{ job_id: string | null; status: string; cached: boolean }> {
    return Promise.resolve({ job_id: null, status: "SUCCEEDED", cached: true });
  }
  getCandidateTransformationAnalysis() {
    return Promise.resolve({ recommended_strategies: [{}], eligibility_outcome: "ELIGIBLE" });
  }
  queueTransformationPlans(): Promise<{ job_id: string | null; status: string; cached: boolean }> {
    return Promise.resolve({ job_id: null, status: "SUCCEEDED", cached: true });
  }
  getCandidateTransformationPlans() {
    return Promise.resolve({ plans: [{}], planning_outcome: "PLANNED" });
  }
  queueTransformationGovernance(): Promise<{ job_id: string | null; status: string; cached: boolean }> {
    return Promise.resolve({ job_id: null, status: "SUCCEEDED", cached: true });
  }
  getCandidateTransformationGovernance(): Promise<unknown> {
    return Promise.resolve({});
  }
  createTransformationSelection() {
    return Promise.resolve({
      selected_plan_id: "plan-1",
      effective: true,
      selection_reason_codes: [],
      status: "PLAN_SELECTED"
    });
  }
  createRenderContract() {
    return Promise.resolve({
      contract_ready: true,
      effective: true,
      reason_codes: [],
      status: "EXECUTABLE"
    });
  }
  queueVisualComposition(): Promise<{ job_id: string | null; status: string; cached: boolean }> {
    return Promise.resolve({ job_id: null, status: "SUCCEEDED", cached: true });
  }
  getCandidateVisualComposition() {
    return Promise.resolve({ plan_ready: true, reason_codes: [], status: "READY" });
  }
  queueMaterialization(): Promise<MaterializationQueueResponse> {
    this.submissionStarted();
    const response: MaterializationQueueResponse = {
      job_id: "job-prep",
      status: "QUEUED",
      queued: true,
      cached: false,
      active: true,
      materialization_id: this.materializationId,
      render_execution_id: null,
      phase: "PREPARE",
      lifecycle: "QUEUED",
      semantic_outcome: null,
      human_review_required: false,
      publication_ready: false,
      review_ready: false,
      reasons: [],
      skipped_reason: null
    };
    if (!this.deferSubmission) return Promise.resolve(response);
    return new Promise((resolve) => {
      this.resolveSubmission = resolve;
    });
  }
  resolveQueueSubmission(response: MaterializationQueueResponse): void {
    this.resolveSubmission?.(response);
  }
  getMaterialization(id: string): Promise<Materialization> {
    this.getMaterializationCalls.push(id);
    return Promise.resolve(this.next());
  }
  resumeMaterialization(id: string): Promise<MaterializationQueueResponse> {
    this.resumeCalls.push(id);
    return Promise.resolve({
      job_id: null,
      status: "QUEUED",
      queued: false,
      cached: false,
      active: true,
      materialization_id: id,
      render_execution_id: null,
      phase: "FINALIZE",
      lifecycle: "AWAITING_RENDER",
      semantic_outcome: null,
      human_review_required: false,
      publication_ready: false,
      review_ready: false,
      reasons: [],
      skipped_reason: null
    });
  }
  cancelMaterialization(id: string): Promise<Materialization> {
    this.cancelCalls.push(id);
    this.cancelAttempts += 1;
    if (this.cancelAttempts <= this.cancelFailures) {
      return Promise.reject(new Error("cancel unavailable"));
    }
    return Promise.resolve(materialization({ id, lifecycle: "CANCELLED" }));
  }
  cancelJob(id: string): Promise<Job> {
    this.cancelCalls.push(id);
    return Promise.resolve({ ...job, id, status: "CANCELLED" });
  }
  setCancelFailures(count: number): void {
    this.cancelFailures = count;
    this.cancelAttempts = 0;
  }
  materializationArtifactUrl(id: string): string {
    return `/api/materializations/${id}/artifact`;
  }
}

function callbacks(record: {
  results: number[];
  materializations: Materialization[];
  steps: string[];
}): ProduceCallbacks {
  return {
    onStep: (key, state) => record.steps.push(`${key}:${state}`),
    onStatus: () => undefined,
    onMaterialization: (value) => {
      if (value) {
        record.materializations.push(value);
        record.results.push(value.readiness.review_ready ? 1 : 0);
      }
    },
    onResult: () => record.results.push(2),
    sleep: () => Promise.resolve()
  };
}

async function runToCompletion(api: FakeApi): Promise<{
  run: ProduceRun;
  outcome: Awaited<ProduceRun["outcome"]>;
  record: { results: number[]; materializations: Materialization[]; steps: string[] };
}> {
  const record = { results: [] as number[], materializations: [] as Materialization[], steps: [] as string[] };
  const run = runProduceChain(api, "https://example.org/v", "UNKNOWN" as RightsStatus, callbacks(record));
  const outcome = await run.outcome;
  return { run, outcome, record };
}

describe("runProduceChain Stage 6 state machine", () => {
  it("does not finish at preparation-job success; results require committed readiness", async () => {
    const api = new FakeApi([
      materialization({ lifecycle: "AWAITING_RENDER", artifact_available: true, readiness: { review_ready: false, reasons: ["LIFECYCLE_NOT_COMPLETE"], publication_ready: false } }),
      materialization({ lifecycle: "AWAITING_RENDER", artifact_available: true, readiness: { review_ready: false, reasons: ["MISSING_FINAL_ASSESSMENT"], publication_ready: false } }),
      materialization({
        lifecycle: "COMPLETE",
        semantic_outcome: "READY_FOR_HUMAN_REVIEW",
        human_review_required: true,
        artifact_available: true,
        readiness: { review_ready: true, reasons: [], publication_ready: false }
      })
    ]);
    const { outcome, record } = await runToCompletion(api);
    expect(outcome.kind).toBe("ready");
    // Polled the read-only endpoint multiple times across the pending states.
    expect(api.getMaterializationCalls.length).toBeGreaterThanOrEqual(3);
    expect(api.getMaterializationCalls.every((id) => id === "mat-1")).toBe(true);
    // Result marker 2 appears only once, after the review-ready observation (1).
    expect(record.results.filter((value) => value === 2)).toHaveLength(1);
    expect(record.results.indexOf(2)).toBeGreaterThan(record.results.indexOf(1));
  });

  it("does not offer media merely because artifact_available is true", async () => {
    const api = new FakeApi([
      materialization({ lifecycle: "AWAITING_RENDER", artifact_available: true }),
      materialization({
        lifecycle: "COMPLETE",
        semantic_outcome: "DEFERRED_EVIDENCE",
        artifact_available: true,
        readiness: { review_ready: false, reasons: ["MISSING_FINAL_ASSESSMENT"], publication_ready: false }
      })
    ]);
    const { outcome, record } = await runToCompletion(api);
    expect(outcome.kind).toBe("deferred");
    expect(record.results).not.toContain(2);
  });

  it("tracks the durable id across changing render identities", async () => {
    const api = new FakeApi([
      materialization({ lifecycle: "AWAITING_RENDER", render_execution_id: "render-1" }),
      materialization({ lifecycle: "RUNNING", render_execution_id: "render-2" }),
      materialization({
        lifecycle: "COMPLETE",
        render_execution_id: "render-2",
        semantic_outcome: "READY_FOR_HUMAN_REVIEW",
        human_review_required: true,
        artifact_available: true,
        readiness: { review_ready: true, reasons: [], publication_ready: false }
      })
    ]);
    const { outcome } = await runToCompletion(api);
    expect(outcome.kind).toBe("ready");
    expect(new Set(api.getMaterializationCalls)).toEqual(new Set(["mat-1"]));
  });

  it("bounds resume recovery for a stalled pending generation", async () => {
    const pending = materialization({ lifecycle: "AWAITING_RENDER", render_execution_id: "render-1" });
    const api = new FakeApi([pending]);
    const record = { results: [] as number[], materializations: [] as Materialization[], steps: [] as string[] };
    const callbacksWithClock: ProduceCallbacks = {
      ...callbacks(record),
      sleep: () => Promise.resolve()
    };
    const run = runProduceChain(api, "https://example.org/v", "UNKNOWN" as RightsStatus, callbacksWithClock);
    // Let the chain reach the poll loop, then stop it after enough polls.
    for (let index = 0; index < 8; index += 1) {
      await Promise.resolve();
    }
    await run.requestStop();
    const outcome = await run.outcome;
    expect(outcome.kind).toBe("cancelled");
    expect(api.resumeCalls.length).toBeLessThanOrEqual(1);
  });
});

describe("runProduceChain cancellation", () => {
  it("reconciles an in-flight submission and cancels the exact returned identity", async () => {
    const api = new FakeApi([materialization({ lifecycle: "AWAITING_RENDER" })]);
    api.deferSubmission = true;
    const record = { results: [] as number[], materializations: [] as Materialization[], steps: [] as string[] };
    const run = runProduceChain(api, "https://example.org/v", "UNKNOWN" as RightsStatus, callbacks(record));
    await api.submissionReady;

    // Cancel while queueMaterialization is still pending.
    const stop = run.requestStop();
    api.resolveQueueSubmission({
      job_id: "job-prep",
      status: "QUEUED",
      queued: true,
      cached: false,
      active: true,
      materialization_id: "mat-1",
      render_execution_id: null,
      phase: "PREPARE",
      lifecycle: "QUEUED",
      semantic_outcome: null,
      human_review_required: false,
      publication_ready: false,
      review_ready: false,
      reasons: [],
      skipped_reason: null
    });
    const cancelOutcome = await stop;
    expect(cancelOutcome).toEqual({ status: "confirmed" });
    expect(api.cancelCalls).toEqual(["mat-1"]);
    const outcome = await run.outcome;
    expect(outcome.kind).toBe("cancelled");
  });

  it("surfaces an unconfirmed cancellation and allows a truthful retry", async () => {
    const api = new FakeApi([materialization({ lifecycle: "AWAITING_RENDER" })]);
    api.deferSubmission = true;
    const record = { results: [] as number[], materializations: [] as Materialization[], steps: [] as string[] };
    const run = runProduceChain(api, "https://example.org/v", "UNKNOWN" as RightsStatus, callbacks(record));
    await api.submissionReady;
    api.setCancelFailures(1);

    const first = run.requestStop();
    api.resolveQueueSubmission({
      job_id: "job-prep",
      status: "QUEUED",
      queued: true,
      cached: false,
      active: true,
      materialization_id: "mat-1",
      render_execution_id: null,
      phase: "PREPARE",
      lifecycle: "QUEUED",
      semantic_outcome: null,
      human_review_required: false,
      publication_ready: false,
      review_ready: false,
      reasons: [],
      skipped_reason: null
    });
    const firstOutcome = await first;
    expect(firstOutcome.status).toBe("unconfirmed");

    const secondOutcome = await run.requestStop();
    expect(secondOutcome).toEqual({ status: "confirmed" });
    expect(api.cancelCalls).toEqual(["mat-1", "mat-1"]);
  });
});
