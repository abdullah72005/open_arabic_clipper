"use client";

import { useEffect, useRef, useState } from "react";

import {
  api,
  ApiError,
  type Job,
  type RightsStatus
} from "@/lib/api-client";
import {
  initialStepStates,
  isActiveJobStatus,
  isTerminalJobStatus,
  pickEffectiveRefinement,
  PRODUCE_STEPS,
  selectTopCandidate,
  type ProduceStepKey,
  type StepStates
} from "@/lib/produce";

const POLL_INTERVAL_MS = 2500;
const CHAIN_TIMEOUT_MS = 45 * 60 * 1000;

const rightsOptions: Array<{ value: RightsStatus; label: string }> = [
  { value: "UNKNOWN", label: "Unknown — process locally; publishing requires review" },
  { value: "OWNED", label: "Owned" },
  { value: "LICENSED", label: "Licensed" },
  { value: "PERMISSION", label: "Permission granted" },
  { value: "PUBLIC_DOMAIN", label: "Public domain" },
  { value: "OTHER_ALLOWED", label: "Other allowed use" },
  { value: "THIRD_PARTY_REUSE", label: "Third-party — reuse claimed" },
  { value: "THIRD_PARTY_UNKNOWN", label: "Third-party — rights unknown" }
];

interface QueuedStage {
  job_id: string | null;
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

function assertNotAborted(signal: AbortSignal): void {
  if (signal.aborted) throw new DOMException("Aborted", "AbortError");
}

export function ProduceRunner() {
  const [url, setUrl] = useState("");
  const [rightsStatus, setRightsStatus] = useState<RightsStatus>("UNKNOWN");
  const [steps, setSteps] = useState<StepStates>(initialStepStates);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [result, setResult] = useState<{ id: string; url: string } | null>(null);
  const [running, setRunning] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  const run = async () => {
    const trimmed = url.trim();
    if (!trimmed) {
      setError("Paste a public, permitted media link first.");
      return;
    }
    const controller = new AbortController();
    abortRef.current = controller;
    const signal = controller.signal;

    setRunning(true);
    setError("");
    setResult(null);
    setSteps(initialStepStates());

    let currentKey: ProduceStepKey | null = null;
    const setStep = (key: ProduceStepKey, state: StepStates[ProduceStepKey]) =>
      setSteps((previous) => ({ ...previous, [key]: state }));
    const activate = (key: ProduceStepKey) => {
      currentKey = key;
      setStep(key, "active");
    };

    const waitForJob = async (jobId: string): Promise<Job> => {
      for (;;) {
        assertNotAborted(signal);
        const job = await api.getJob(jobId);
        if (isTerminalJobStatus(job.status)) return job;
        await sleep(POLL_INTERVAL_MS, signal);
      }
    };

    const runQueued = async (stage: QueuedStage): Promise<void> => {
      if (!stage.job_id) return;
      const job = await waitForJob(stage.job_id);
      if (job.status !== "SUCCEEDED") {
        throw new Error(job.error_message ?? `Job ${job.status.toLowerCase()}.`);
      }
    };

    const waitForCandidates = async (sourceId: string): Promise<void> => {
      const started = Date.now();
      let sawJob = false;
      for (;;) {
        assertNotAborted(signal);
        const jobs = (await api.listJobs()).filter((job) => job.source_video_id === sourceId);
        if (jobs.length > 0) sawJob = true;
        const active = jobs.filter((job) => isActiveJobStatus(job.status));
        const newest = [...jobs].sort(
          (left, right) => Date.parse(right.created_at) - Date.parse(left.created_at)
        )[0];
        const observed = active[0] ?? newest;
        if (observed) setStatus(`Processing: ${observed.kind} (${observed.status.toLowerCase()})`);
        if (active.length === 0 && sawJob) {
          try {
            await api.getCandidateAnalysis(sourceId);
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
        await sleep(POLL_INTERVAL_MS, signal);
      }
    };

    try {
      activate("ingest");
      setStatus("Submitting source…");
      const sources = await api.submitUrls([trimmed], rightsStatus);
      const source = sources[0];
      if (!source) throw new Error("The source was not created.");
      setStatus("Ingesting, transcribing, and finding candidates…");
      await waitForCandidates(source.id);
      setStep("ingest", "done");

      const candidate = selectTopCandidate(await api.listCandidates(source.id, { limit: 100 }));
      if (!candidate) throw new Error("No retained clip candidate was found for this source.");
      setStatus(
        `Selected candidate ${candidate.candidate_key} (score ${Math.round(candidate.clip_score * 100)}%).`
      );

      activate("refine");
      setStatus("Refining transcript at final-clip quality…");
      await runQueued(await api.queueCandidateRefinement(candidate.id, "FINAL_CLIP"));
      let refinements = await api.listCandidateRefinements(candidate.id);
      let effective = pickEffectiveRefinement(refinements);
      if (!effective) {
        setStatus("Final refinement is unresolved; running candidate-grade refinement…");
        await runQueued(await api.queueCandidateRefinement(candidate.id, "CANDIDATE"));
        refinements = await api.listCandidateRefinements(candidate.id);
        effective = pickEffectiveRefinement(refinements);
      }
      if (!effective) throw new Error("No usable transcript refinement was produced.");
      setStep("refine", "done");

      activate("analyze");
      setStatus("Analyzing transformation eligibility (Stage 4.0)…");
      await runQueued(await api.queueTransformationAnalysis(candidate.id));
      const analysis = await api.getCandidateTransformationAnalysis(candidate.id);
      if (analysis.recommended_strategies.length === 0) {
        throw new Error(
          `No transformation strategy is recommended (${analysis.eligibility_outcome ?? "unknown"}).`
        );
      }
      setStep("analyze", "done");

      activate("plan");
      setStatus("Generating a transformation plan (Stage 4.1)…");
      await runQueued(await api.queueTransformationPlans(candidate.id));
      const planSet = await api.getCandidateTransformationPlans(candidate.id);
      if (planSet.plans.length === 0) {
        throw new Error(`No transformation plan was produced (${planSet.planning_outcome ?? "none"}).`);
      }
      setStep("plan", "done");

      activate("govern");
      setStatus("Governing the plan (Stage 4.2)…");
      await runQueued(await api.queueTransformationGovernance(candidate.id));
      await api.getCandidateTransformationGovernance(candidate.id);
      setStep("govern", "done");

      activate("select");
      setStatus("Selecting the final plan (Stage 4.3)…");
      const selection = await api.createTransformationSelection(candidate.id);
      if (!selection.selected_plan_id || !selection.effective) {
        throw new Error(
          `No final plan could be selected (${selection.selection_reason_codes.join(", ") || selection.status}).`
        );
      }
      setStep("select", "done");

      activate("contract");
      setStatus("Building the render contract (Stage 5.0)…");
      const contract = await api.createRenderContract(candidate.id);
      if (!contract.contract_ready || !contract.effective) {
        throw new Error(
          `The render contract is not executable (${contract.reason_codes.join(", ") || contract.status}).`
        );
      }
      setStep("contract", "done");

      activate("compose");
      setStatus("Composing framing, captions, and overlays (Stage 5.1)…");
      await runQueued(await api.queueVisualComposition(candidate.id));
      const composition = await api.getCandidateVisualComposition(candidate.id);
      if (!composition.plan_ready) {
        throw new Error(
          `Visual composition is not ready (${composition.reason_codes.join(", ") || composition.status}).`
        );
      }
      setStep("compose", "done");

      activate("render");
      setStatus("Rendering the final clip (Stage 5.2)…");
      await runQueued(await api.queueRenderExecution(candidate.id));
      const render = await api.getCandidateRenderExecution(candidate.id);
      if (render.lifecycle !== "COMPLETE" || !render.cache_eligible) {
        throw new Error(render.error_code ?? `The render did not complete (${render.lifecycle}).`);
      }
      setStep("render", "done");
      setResult({ id: render.id, url: api.renderExecutionArtifactUrl(render.id) });
      setStatus("Final clip ready.");
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === "AbortError") {
        setStatus("Cancelled.");
      } else {
        if (currentKey) setStep(currentKey, "error");
        setError(cause instanceof Error ? cause.message : "Production failed.");
        setStatus("");
      }
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  };

  return (
    <>
      <form
        className="card"
        onSubmit={(event) => {
          event.preventDefault();
          if (!running) void run();
        }}
      >
        <label>
          Media link (public and permitted)
          <input
            disabled={running}
            onChange={(event) => setUrl(event.target.value)}
            placeholder="https://example.org/video"
            type="url"
            value={url}
          />
        </label>
        <label>
          Rights status
          <select
            disabled={running}
            onChange={(event) => setRightsStatus(event.target.value as RightsStatus)}
            value={rightsStatus}
          >
            {rightsOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <p>
          <button className="button" disabled={running} type="submit">
            {running ? "Producing…" : "Produce final clip"}
          </button>
          {running && (
            <button
              className="button danger"
              onClick={() => abortRef.current?.abort()}
              style={{ marginLeft: 8 }}
              type="button"
            >
              Cancel
            </button>
          )}
        </p>
      </form>

      <section className="card">
        <h3>Steps</h3>
        <ol className="produce-steps">
          {PRODUCE_STEPS.map((step) => (
            <li className={`produce-step ${steps[step.key]}`} key={step.key}>
              <span aria-hidden className="produce-step-marker">
                {steps[step.key] === "done" ? "✓" : steps[step.key] === "active" ? "…" : steps[step.key] === "error" ? "!" : "•"}
              </span>
              <span>
                <strong>{step.label}</strong>
                <span className="muted"> — {step.detail}</span>
              </span>
            </li>
          ))}
        </ol>
        {status && <p className="muted">{status}</p>}
        {error && <p className="error">{error}</p>}
      </section>

      {result && (
        <section className="card">
          <h3>Final result</h3>
          <video controls playsInline src={result.url} style={{ width: "100%", borderRadius: 12 }} />
          <p>
            <a className="button" download href={result.url}>
              Download MP4
            </a>
          </p>
          <p className="muted">
            This is the Stage 5.2 rendered clip for the selected candidate. Publication eligibility is
            evaluated separately from transcript readiness.
          </p>
        </section>
      )}
    </>
  );
}
