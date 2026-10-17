"use client";

import Link from "next/link";
import React, { useEffect, useRef, useState } from "react";

import { api, type Materialization, type RightsStatus } from "@/lib/api-client";
import {
  initialStepStates,
  PRODUCE_STEPS,
  type ProduceStepKey,
  type StepStates
} from "@/lib/produce";
import {
  runProduceChain,
  type ProduceOutcome,
  type ProduceRun
} from "@/lib/produce-runner";

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

type CancelState =
  | { status: "requesting" }
  | { status: "confirmed" }
  | { status: "unconfirmed"; error: string };

function describeOutcome(outcome: ProduceOutcome): string {
  switch (outcome.kind) {
    case "deferred":
      return `Deferred: ${outcome.detail}`;
    case "rejected":
      return `Rejected by quality assessment: ${outcome.detail}`;
    case "blocked":
      return `Blocked: ${outcome.detail}`;
    case "stale":
      return `Stale generation — upstream revalidation required: ${outcome.detail}`;
    case "unavailable":
      return `Assessment unavailable: ${outcome.detail}`;
    case "diagnostic":
      return `No durable materialization to inspect: ${outcome.detail}`;
    case "failed":
      return `Failed: ${outcome.detail}`;
    case "no_output":
      return "No output was produced — a valid zero-output result.";
    default:
      return "";
  }
}

export function ProduceRunner() {
  const [url, setUrl] = useState("");
  const [rightsStatus, setRightsStatus] = useState<RightsStatus>("UNKNOWN");
  const [steps, setSteps] = useState<StepStates>(initialStepStates);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [result, setResult] = useState<{ id: string; url: string } | null>(null);
  const [materialization, setMaterialization] = useState<Materialization | null>(null);
  const [running, setRunning] = useState(false);
  const [cancelState, setCancelState] = useState<CancelState | null>(null);
  const runRef = useRef<ProduceRun | null>(null);
  const runTokenRef = useRef<symbol | null>(null);

  useEffect(() => () => void runRef.current?.requestStop(), []);

  // Cancel is server-authoritative: the run reconciles any in-flight submission,
  // then cancels the exact returned materialization/job before declaring server
  // cancellation complete. A failure is surfaced as unconfirmed with a retry.
  const cancel = async () => {
    const run = runRef.current;
    if (!run) {
      setCancelState({ status: "confirmed" });
      return;
    }
    setCancelState({ status: "requesting" });
    const outcome = await run.requestStop();
    if (outcome.status === "unconfirmed") {
      setCancelState({ status: "unconfirmed", error: outcome.error });
      return;
    }
    setCancelState({ status: "confirmed" });
    setStatus("Cancelled.");
  };

  const run = () => {
    const trimmed = url.trim();
    if (!trimmed) {
      setError("Paste a public, permitted media link first.");
      return;
    }
    // Never start another Produce run while a previous run's cancellation is
    // still unresolved: its controller must own reconciliation to completion.
    if (running) return;
    setRunning(true);
    setError("");
    setStatus("");
    setResult(null);
    setMaterialization(null);
    setCancelState(null);
    setSteps(initialStepStates());

    // A per-run identity token guards every callback against an obsolete run
    // updating a newer one (the callbacks can fire synchronously before the run
    // object itself is assigned).
    const token = Symbol("produce-run");
    runTokenRef.current = token;
    const isCurrent = () => runTokenRef.current === token;
    const produceRun = runProduceChain(api, trimmed, rightsStatus, {
      onStep: (key: ProduceStepKey, state) => {
        if (isCurrent()) setSteps((p) => ({ ...p, [key]: state }));
      },
      onStatus: (text) => {
        if (isCurrent()) setStatus(text);
      },
      onMaterialization: (value) => {
        if (isCurrent()) setMaterialization(value);
      },
      onResult: (value) => {
        if (isCurrent()) setResult(value);
      }
    });
    runRef.current = produceRun;
    produceRun.outcome
      .then((outcome) => {
        if (!isCurrent()) return;
        if (outcome.kind === "ready" || outcome.kind === "cancelled") {
          if (outcome.kind === "cancelled") setStatus("Cancelled.");
          return;
        }
        if (outcome.kind === "failed") {
          setError(describeOutcome(outcome));
          setStatus("");
          return;
        }
        setStatus(describeOutcome(outcome));
      })
      .catch((cause) => {
        if (!isCurrent()) return;
        setError(cause instanceof Error ? cause.message : "Production failed.");
        setStatus("");
      })
      .finally(() => {
        // The controller is retained until the outcome settles, i.e. until server
        // cancellation is acknowledged or authoritative reads establish the work
        // already terminated. Only then is it safe to clear and re-enable Produce.
        if (!isCurrent()) return;
        runTokenRef.current = null;
        setRunning(false);
        runRef.current = null;
      });
  };

  return (
    <>
      <form
        className="card"
        onSubmit={(event) => {
          event.preventDefault();
          if (!running) run();
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
          {running && cancelState?.status !== "confirmed" && cancelState?.status !== "unconfirmed" && (
            <button
              className="button danger"
              disabled={cancelState?.status === "requesting"}
              onClick={() => void cancel()}
              style={{ marginLeft: 8 }}
              type="button"
            >
              {cancelState?.status === "requesting" ? "Cancelling…" : "Cancel"}
            </button>
          )}
        </p>
        {cancelState?.status === "unconfirmed" && (
          <p className="error">
            Cancellation could not be confirmed: {cancelState.error}. Server work may still be
            running.{" "}
            <button className="button danger" onClick={() => void cancel()} type="button">
              Retry cancel
            </button>
          </p>
        )}
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

      {materialization && (
        <section className="card">
          <h3>Transformed short (Stage 6)</h3>
          {result && (
            <video
              controls
              playsInline
              src={result.url}
              style={{ width: "100%", borderRadius: 12 }}
            />
          )}
          <p>
            <strong>Lifecycle:</strong> {materialization.lifecycle}
            {materialization.semantic_outcome ? ` · ${materialization.semantic_outcome}` : ""}
          </p>
          <p>
            <strong>Review:</strong>{" "}
            {materialization.readiness.review_ready
              ? "ready for human review"
              : `not ready (${materialization.readiness.reasons.join(", ") || materialization.live_freshness})`}
          </p>
          <p className="muted">
            Materialization {materialization.id.slice(0, 8)} · render{" "}
            {materialization.render_execution_id?.slice(0, 8) ?? "—"} · output{" "}
            {materialization.output_fingerprint.slice(0, 12)} · publication_ready=false
          </p>
          {materialization.findings.length > 0 && (
            <ul className="muted">
              {materialization.findings.slice(0, 8).map((finding, index) => (
                <li key={index}>
                  {String(finding.dimension ?? "finding")}: {String(finding.level ?? "")}
                </li>
              ))}
            </ul>
          )}
          <p>
            {result ? (
              <a className="button" download href={result.url}>
                Download transformed MP4
              </a>
            ) : (
              <span className="muted">
                {materialization.artifact_available
                  ? "Media exists but is not review-ready; no transformed short is offered."
                  : "No verified transformed artifact is available."}
              </span>
            )}{" "}
            <Link className="button" href={`/sources/${materialization.source_video_id}`}>
              Source
            </Link>
          </p>
          <p className="muted">
            Publication eligibility is evaluated separately from transcript readiness.
          </p>
        </section>
      )}
    </>
  );
}
