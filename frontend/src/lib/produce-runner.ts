import type {
  Candidate,
  CandidateRefinement,
  Job,
  Materialization,
  MaterializationQueueResponse,
  RightsStatus,
  Source
} from "@/lib/api-client";
import { ApiError } from "@/lib/api-client";
import {
  isActiveJobStatus,
  isTerminalJobStatus,
  pickEffectiveRefinement,
  selectTopCandidate,
  type ProduceStepKey,
  type StepState
} from "@/lib/produce";

export const POLL_INTERVAL_MS = 2500;
export const CHAIN_TIMEOUT_MS = 45 * 60 * 1000;
//: Polls of an unchanged AWAITING_RENDER render identity before one bounded
//: durable-resume recovery is attempted (never a repeated resume loop).
export const RECOVERY_STALL_POLLS = 4;
export const MAX_RESUME_ATTEMPTS = 1;

export interface TransformationAnalysisView {
  recommended_strategies: unknown[];
  eligibility_outcome: string | null;
}

export interface PlanSetView {
  plans: unknown[];
  planning_outcome: string | null;
}

export interface SelectionView {
  selected_plan_id: string | null;
  effective: boolean;
  selection_reason_codes: string[];
  status: string;
}

export interface ContractView {
  contract_ready: boolean;
  effective: boolean;
  reason_codes: string[];
  status: string;
}

export interface VisualCompositionView {
  plan_ready: boolean;
  reason_codes: string[];
  status: string;
}

export interface QueuedJob {
  job_id: string | null;
  status: string;
  cached: boolean;
}

/**
 * The exact subset of the API client the Produce chain uses. Structural typing
 * lets the real `api` client satisfy it; tests supply deterministic fakes.
 */
export interface ProduceApi {
  submitUrls(urls: string[], rightsStatus: RightsStatus): Promise<Source[]>;
  listJobs(): Promise<Job[]>;
  getJob(id: string): Promise<Job>;
  getCandidateAnalysis(sourceId: string): Promise<unknown>;
  listCandidates(sourceId: string, options?: { limit?: number }): Promise<Candidate[]>;
  queueCandidateRefinement(
    candidateId: string,
    priority: "CANDIDATE" | "FINAL_CLIP"
  ): Promise<QueuedJob>;
  listCandidateRefinements(candidateId: string): Promise<CandidateRefinement[]>;
  queueTransformationAnalysis(candidateId: string): Promise<QueuedJob>;
  getCandidateTransformationAnalysis(candidateId: string): Promise<TransformationAnalysisView>;
  queueTransformationPlans(candidateId: string): Promise<QueuedJob>;
  getCandidateTransformationPlans(candidateId: string): Promise<PlanSetView>;
  queueTransformationGovernance(candidateId: string): Promise<QueuedJob>;
  getCandidateTransformationGovernance(candidateId: string): Promise<unknown>;
  createTransformationSelection(candidateId: string): Promise<SelectionView>;
  createRenderContract(candidateId: string): Promise<ContractView>;
  queueVisualComposition(candidateId: string): Promise<QueuedJob>;
  getCandidateVisualComposition(candidateId: string): Promise<VisualCompositionView>;
  queueMaterialization(candidateId: string): Promise<MaterializationQueueResponse>;
  getMaterialization(id: string): Promise<Materialization>;
  resumeMaterialization(id: string): Promise<MaterializationQueueResponse>;
  cancelMaterialization(id: string): Promise<Materialization>;
  cancelJob(id: string): Promise<Job>;
  materializationArtifactUrl(id: string): string;
}

export interface ProduceCallbacks {
  onStep(key: ProduceStepKey, state: StepState): void;
  onStatus(text: string): void;
  onMaterialization(value: Materialization | null): void;
  onResult(value: { id: string; url: string } | null): void;
  sleep?: (ms: number) => Promise<void>;
}

export type ProduceOutcomeKind =
  | "ready"
  | "no_output"
  | "deferred"
  | "rejected"
  | "blocked"
  | "cancelled"
  | "failed"
  | "stale"
  | "unavailable"
  | "diagnostic";

export interface ProduceOutcome {
  kind: ProduceOutcomeKind;
  detail: string;
}

export type CancelOutcome =
  | { status: "confirmed" }
  | { status: "no_active_work" }
  | { status: "unconfirmed"; error: string };

/**
 * Per-run controller disposition. ``RUNNING`` is active work; ``STOPPING`` means
 * local stop intent has been expressed but server cancellation is being
 * reconciled; ``CANCEL_UNCONFIRMED`` means a cancellation attempt failed and must
 * be retried (or resolved by authoritative reads); ``TERMINAL`` means the run has
 * settled and no further reconciliation is required.
 */
export type RunState = "RUNNING" | "STOPPING" | "CANCEL_UNCONFIRMED" | "TERMINAL";

export interface ProduceRun {
  readonly state: RunState;
  readonly outcome: Promise<ProduceOutcome>;
  readonly requestStop: () => Promise<CancelOutcome>;
}

class StopRequested extends Error {
  constructor() {
    super("Produce stop requested");
    this.name = "StopRequested";
  }
}

function errorMessage(cause: unknown): string {
  return cause instanceof Error ? cause.message : "Unknown error";
}

function defaultSleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isTerminalLifecycle(lifecycle: string): boolean {
  return lifecycle === "CANCELLED" || lifecycle === "COMPLETE" || lifecycle === "FAILED" ||
    lifecycle === "BLOCKED";
}

function isTerminalJob(status: string): boolean {
  return status === "CANCELLED" || status === "SUCCEEDED" || status === "FAILED";
}

/**
 * Run the single-candidate Produce chain (ingest → Stage 4 → Stage 5 → Stage 6).
 *
 * Stage 6 tracks the durable materialization ID across preparation, rendering,
 * callback/finalization, and assessment: the preparation job's success is only
 * intermediate progress, and the read-only materialization endpoint is polled
 * across changing render/finalizer job identities. The transformed result is
 * exposed only after the authoritative committed readiness response permits
 * review (``readiness.review_ready`` and a verified artifact); ``artifact_available``
 * alone never suffices. Blocked, deferred, rejected, cancelled, stale, no-output,
 * and unavailable outcomes are surfaced explicitly.
 *
 * Cancellation is owned by one per-run controller. Local stop intent
 * (``STOPPING``) is never reported as server cancellation: ``outcome`` settles to
 * ``cancelled`` only when server cancellation is acknowledged, no active work
 * remains, or authoritative reads establish the run's work already terminated.
 * Every in-flight create/queue operation (including pre-Stage-6 ones) is tracked
 * and reconciled, so a late-returned identity is cancelled before declaring "no
 * active work". An unconfirmed cancellation keeps the controller alive
 * (``CANCEL_UNCONFIRMED``) so a retry performs fresh reconciliation rather than a
 * false confirmation.
 */
export function runProduceChain(
  api: ProduceApi,
  url: string,
  rightsStatus: RightsStatus,
  callbacks: ProduceCallbacks
): ProduceRun {
  const sleep = callbacks.sleep ?? defaultSleep;
  let state: RunState = "RUNNING";

  let settleOutcome!: (value: ProduceOutcome) => void;
  let rejectOutcome!: (reason: unknown) => void;
  const outcome = new Promise<ProduceOutcome>((resolve, reject) => {
    settleOutcome = resolve;
    rejectOutcome = reject;
  });

  // Exact identities discovered for THIS run. Only these are ever cancelled.
  let materializationId: string | null = null;
  let activeJobId: string | null = null;
  let pendingQueueJobId: string | null = null;
  let ingestJobId: string | null = null;
  let submittedSourceId: string | null = null;

  const inFlight = new Set<Promise<unknown>>();
  let lastCancel: CancelOutcome | null = null;
  let cancelInFlight: Promise<CancelOutcome> | null = null;

  const checkStop = (): void => {
    if (state !== "RUNNING") throw new StopRequested();
  };

  const track = <T,>(operation: Promise<T>, capture?: (value: T) => void): Promise<T> => {
    const wrapped = operation.then((value) => {
      capture?.(value);
      return value;
    });
    inFlight.add(wrapped);
    const remove = () => inFlight.delete(wrapped);
    void wrapped.then(remove, remove);
    return wrapped;
  };

  const drainInFlight = async (): Promise<void> => {
    // Loop because a drained operation may synchronously register another.
    for (let attempt = 0; attempt < 64; attempt += 1) {
      const pending = [...inFlight];
      if (pending.length === 0) return;
      await Promise.allSettled(pending);
    }
  };

  const captureQueuedJob = (stage: QueuedJob | null | undefined): void => {
    if (stage && stage.job_id) pendingQueueJobId = stage.job_id;
  };

  const lifecycleOutcome = (lifecycle: string): CancelOutcome =>
    isTerminalLifecycle(lifecycle)
      ? { status: "confirmed" }
      : { status: "unconfirmed", error: `materialization ${lifecycle}` };

  const jobOutcome = (status: string): CancelOutcome =>
    isTerminalJob(status) ? { status: "confirmed" } : { status: "unconfirmed", error: `job ${status}` };

  // Resolve the exact active ingest job observed for this submitted source. Never
  // a job that does not belong to this run's source.
  const findActiveIngestJob = async (): Promise<string | null> => {
    if (!submittedSourceId) return null;
    try {
      const jobs = await api.listJobs();
      const active = jobs.find(
        (job) => job.source_video_id === submittedSourceId && isActiveJobStatus(job.status)
      );
      return active ? active.id : null;
    } catch {
      return null;
    }
  };

  const performCancellation = async (): Promise<CancelOutcome> => {
    const id = materializationId;
    if (id) {
      try {
        const row = await api.cancelMaterialization(id);
        return lifecycleOutcome(row.lifecycle);
      } catch (cause) {
        return { status: "unconfirmed", error: errorMessage(cause) };
      }
    }
    let jobId = activeJobId ?? pendingQueueJobId ?? ingestJobId;
    if (!jobId) jobId = await findActiveIngestJob();
    if (jobId) {
      try {
        const job = await api.cancelJob(jobId);
        return jobOutcome(job.status);
      } catch (cause) {
        return { status: "unconfirmed", error: errorMessage(cause) };
      }
    }
    return { status: "no_active_work" };
  };

  const settleCancellation = (result: CancelOutcome): void => {
    if (result.status === "unconfirmed") {
      state = "CANCEL_UNCONFIRMED";
      return;
    }
    state = "TERMINAL";
    lastCancel = result;
    settleOutcome({
      kind: "cancelled",
      detail: result.status === "no_active_work" ? "no active work" : "server cancellation confirmed"
    });
  };

  const requestStop = (): Promise<CancelOutcome> => {
    if (state === "TERMINAL") {
      return Promise.resolve(lastCancel ?? { status: "confirmed" });
    }
    if (state === "RUNNING") state = "STOPPING";
    if (cancelInFlight) return cancelInFlight;
    cancelInFlight = (async () => {
      // Fresh reconciliation on every attempt: await outstanding create/queue
      // operations (capturing late-returned identities) before deciding.
      await drainInFlight();
      const result = await performCancellation();
      settleCancellation(result);
      return result;
    })().finally(() => {
      cancelInFlight = null;
    });
    return cancelInFlight;
  };

  const body = (async (): Promise<void> => {
    const activate = (key: ProduceStepKey) => callbacks.onStep(key, "active");
    const waitForJob = async (jobId: string): Promise<Job> => {
      activeJobId = jobId;
      try {
        for (;;) {
          checkStop();
          const job = await api.getJob(jobId);
          checkStop();
          if (isTerminalJobStatus(job.status)) return job;
          await sleep(POLL_INTERVAL_MS);
        }
      } finally {
        if (activeJobId === jobId) activeJobId = null;
      }
    };
    const queue = async (operation: Promise<QueuedJob>): Promise<QueuedJob> => {
      const stage = await track(operation, captureQueuedJob);
      checkStop();
      return stage;
    };
    const runQueued = async (stage: QueuedJob | null | undefined): Promise<void> => {
      if (!stage || !stage.job_id) return;
      const job = await waitForJob(stage.job_id);
      if (job.status !== "SUCCEEDED") {
        throw new Error(job.error_message ?? `Job ${job.status.toLowerCase()}.`);
      }
    };

    const waitForCandidates = async (sourceId: string): Promise<void> => {
      const started = Date.now();
      let sawJob = false;
      for (;;) {
        checkStop();
        const jobs = (await api.listJobs()).filter((job) => job.source_video_id === sourceId);
        checkStop();
        if (jobs.length > 0) sawJob = true;
        const active = jobs.filter((job) => isActiveJobStatus(job.status));
        const newest = [...jobs].sort(
          (left, right) => Date.parse(right.created_at) - Date.parse(left.created_at)
        )[0];
        const observed = active[0] ?? newest;
        if (observed) {
          ingestJobId = observed.id;
          callbacks.onStatus(`Processing: ${observed.kind} (${observed.status.toLowerCase()})`);
        }
        if (active.length === 0 && sawJob) {
          try {
            await api.getCandidateAnalysis(sourceId);
            checkStop();
            return;
          } catch (cause) {
            if (!(cause instanceof ApiError && cause.status === 404)) throw cause;
            const failed = jobs.find((job) => job.status === "FAILED");
            if (failed) throw new Error(failed.error_message ?? "Source processing failed.");
          }
        }
        if (Date.now() - started > CHAIN_TIMEOUT_MS) {
          throw new Error("Timed out waiting for ingest, transcription, and candidate analysis.");
        }
        await sleep(POLL_INTERVAL_MS);
      }
    };

    try {
      checkStop();
      activate("ingest");
      callbacks.onStatus("Submitting source…");
      const sources = await track(api.submitUrls([url], rightsStatus), (created) => {
        if (created[0]) submittedSourceId = created[0].id;
      });
      checkStop();
      const source = sources[0];
      if (!source) throw new Error("The source was not created.");
      callbacks.onStatus("Ingesting, transcribing, and finding candidates…");
      await waitForCandidates(source.id);
      callbacks.onStep("ingest", "done");

      const candidate = selectTopCandidate(await api.listCandidates(source.id, { limit: 100 }));
      checkStop();
      if (!candidate) throw new Error("No retained clip candidate was found for this source.");
      callbacks.onStatus(
        `Selected candidate ${candidate.candidate_key} (score ${Math.round(candidate.clip_score * 100)}%).`
      );

      activate("refine");
      callbacks.onStatus("Refining transcript at final-clip quality…");
      await runQueued(await queue(api.queueCandidateRefinement(candidate.id, "FINAL_CLIP")));
      let refinements = await api.listCandidateRefinements(candidate.id);
      checkStop();
      let effective = pickEffectiveRefinement(refinements);
      if (!effective) {
        callbacks.onStatus("Final refinement is unresolved; running candidate-grade refinement…");
        await runQueued(await queue(api.queueCandidateRefinement(candidate.id, "CANDIDATE")));
        refinements = await api.listCandidateRefinements(candidate.id);
        checkStop();
        effective = pickEffectiveRefinement(refinements);
      }
      if (!effective) throw new Error("No usable transcript refinement was produced.");
      callbacks.onStep("refine", "done");

      activate("analyze");
      callbacks.onStatus("Analyzing transformation eligibility (Stage 4.0)…");
      await runQueued(await queue(api.queueTransformationAnalysis(candidate.id)));
      const analysis = await api.getCandidateTransformationAnalysis(candidate.id);
      checkStop();
      if (analysis.recommended_strategies.length === 0) {
        throw new Error(
          `No transformation strategy is recommended (${analysis.eligibility_outcome ?? "unknown"}).`
        );
      }
      callbacks.onStep("analyze", "done");

      activate("plan");
      callbacks.onStatus("Generating a transformation plan (Stage 4.1)…");
      await runQueued(await queue(api.queueTransformationPlans(candidate.id)));
      const planSet = await api.getCandidateTransformationPlans(candidate.id);
      checkStop();
      if (planSet.plans.length === 0) {
        throw new Error(`No transformation plan was produced (${planSet.planning_outcome ?? "none"}).`);
      }
      callbacks.onStep("plan", "done");

      activate("govern");
      callbacks.onStatus("Governing the plan (Stage 4.2)…");
      await runQueued(await queue(api.queueTransformationGovernance(candidate.id)));
      await api.getCandidateTransformationGovernance(candidate.id);
      checkStop();
      callbacks.onStep("govern", "done");

      activate("select");
      callbacks.onStatus("Selecting the final plan (Stage 4.3)…");
      const selection = await api.createTransformationSelection(candidate.id);
      checkStop();
      if (!selection.selected_plan_id || !selection.effective) {
        throw new Error(
          `No final plan could be selected (${selection.selection_reason_codes.join(", ") || selection.status}).`
        );
      }
      callbacks.onStep("select", "done");

      activate("contract");
      callbacks.onStatus("Building the render contract (Stage 5.0)…");
      const contract = await api.createRenderContract(candidate.id);
      checkStop();
      if (!contract.contract_ready || !contract.effective) {
        throw new Error(
          `The render contract is not executable (${contract.reason_codes.join(", ") || contract.status}).`
        );
      }
      callbacks.onStep("contract", "done");

      activate("compose");
      callbacks.onStatus("Composing framing, captions, and overlays (Stage 5.1)…");
      await runQueued(await queue(api.queueVisualComposition(candidate.id)));
      const composition = await api.getCandidateVisualComposition(candidate.id);
      checkStop();
      if (!composition.plan_ready) {
        throw new Error(
          `Visual composition is not ready (${composition.reason_codes.join(", ") || composition.status}).`
        );
      }
      callbacks.onStep("compose", "done");

      activate("materialize");
      callbacks.onStatus("Materializing the transformed short (Stage 6)…");
      checkStop();
      const queued = await track(api.queueMaterialization(candidate.id), (response) => {
        materializationId = response.materialization_id;
        if (response.job_id) pendingQueueJobId = response.job_id;
      });
      checkStop();
      const id = queued.materialization_id;
      if (!id) throw new Error("The materialization was not created.");

      const result = await pollMaterialization(api, id, {
        checkStop,
        sleep,
        onStatus: callbacks.onStatus,
        onMaterialization: callbacks.onMaterialization,
        onResult: callbacks.onResult
      });
      if (result.kind !== "ready" && result.kind !== "cancelled") {
        callbacks.onStep("materialize", "error");
      } else if (result.kind === "ready") {
        callbacks.onStep("materialize", "done");
      }
      settleOutcome(result);
    } catch (cause) {
      if (cause instanceof StopRequested) {
        // Local stop intent is not server cancellation. Leave the outcome pending;
        // the controller resolves it once cancellation is acknowledged (or an
        // authoritative read establishes the work already terminated).
        return;
      }
      rejectOutcome(cause);
    }
  })();
  void body;

  return {
    get state() {
      return state;
    },
    outcome,
    requestStop
  };
}

interface PollOptions {
  checkStop: () => void;
  sleep: (ms: number) => Promise<void>;
  onStatus: (text: string) => void;
  onMaterialization: (value: Materialization | null) => void;
  onResult: (value: { id: string; url: string } | null) => void;
}

/**
 * Poll the durable, read-only materialization endpoint to a truthful terminal
 * outcome. The durable ID is fixed for the whole run; render/finalizer job
 * identities change underneath it and are never assumed. At most one bounded
 * durable-resume recovery is attempted for a stalled pending-finalization
 * generation; no provider retries are ever driven by polling. A stop intent
 * aborts polling (throwing ``StopRequested``) so cancellation ownership stays
 * with the controller and a later result is never exposed.
 */
async function pollMaterialization(
  api: ProduceApi,
  id: string,
  options: PollOptions
): Promise<ProduceOutcome> {
  const started = Date.now();
  let lastRenderId: string | null = null;
  let stalledPolls = 0;
  let resumeAttempts = 0;

  for (;;) {
    options.checkStop();
    const materialization = await api.getMaterialization(id).catch((cause: unknown) => {
      if (cause instanceof ApiError && cause.status === 404) return null;
      throw cause;
    });
    // A stop requested while this GET was in flight must not expose what it returned.
    options.checkStop();
    if (materialization === null) {
      return { kind: "diagnostic", detail: "materialization not found" };
    }
    options.onMaterialization(materialization);

    if (
      materialization.readiness.review_ready &&
      materialization.lifecycle === "COMPLETE" &&
      materialization.is_current &&
      materialization.artifact_available
    ) {
      options.checkStop();
      options.onResult({ id, url: api.materializationArtifactUrl(id) });
      options.onStatus("Transformed short ready for human review.");
      return { kind: "ready", detail: "" };
    }

    const reasons =
      materialization.readiness.reasons.join(", ") ||
      materialization.reason_codes.join(", ") ||
      materialization.live_freshness;
    if (materialization.lifecycle === "CANCELLED") {
      return { kind: "cancelled", detail: reasons };
    }
    if (materialization.lifecycle === "BLOCKED") {
      return { kind: "blocked", detail: reasons };
    }
    if (materialization.lifecycle === "FAILED") {
      return { kind: "failed", detail: reasons };
    }
    if (materialization.lifecycle === "COMPLETE") {
      const semantic = materialization.semantic_outcome;
      if (semantic === "NO_OUTPUT") return { kind: "no_output", detail: reasons };
      if (semantic === "REJECTED_QUALITY") return { kind: "rejected", detail: reasons };
      if (semantic === "FAILED") return { kind: "failed", detail: reasons };
      if (!materialization.is_current) return { kind: "stale", detail: reasons };
      return {
        kind: "deferred",
        detail: `${semantic ?? "COMPLETE"}${reasons ? `: ${reasons}` : ""}`
      };
    }

    // Pending (QUEUED/RUNNING/AWAITING_RENDER): bounded recovery if the pending
    // render identity has not advanced, so a lost callback can be re-driven once.
    if (materialization.render_execution_id === lastRenderId) {
      stalledPolls += 1;
    } else {
      stalledPolls = 0;
      lastRenderId = materialization.render_execution_id;
    }
    if (stalledPolls >= RECOVERY_STALL_POLLS && resumeAttempts < MAX_RESUME_ATTEMPTS) {
      resumeAttempts += 1;
      stalledPolls = 0;
      try {
        await api.resumeMaterialization(id);
      } catch {
        // The authoritative state is reported by the next read; a conflict is not
        // retried in a loop.
      }
      options.checkStop();
    }

    if (Date.now() - started > CHAIN_TIMEOUT_MS) {
      return { kind: "unavailable", detail: "timed out waiting for Stage 6 assessment" };
    }
    options.onStatus(
      `Stage 6: ${materialization.lifecycle}${
        materialization.semantic_outcome ? ` (${materialization.semantic_outcome})` : ""
      }…`
    );
    await options.sleep(POLL_INTERVAL_MS);
  }
}
