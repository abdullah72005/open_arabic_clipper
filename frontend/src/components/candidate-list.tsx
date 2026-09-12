"use client";

import React, { useCallback, useState } from "react";

import { ApiState } from "@/components/api-state";
import { MixedDirectionText } from "@/components/mixed-direction-text";
import {
  api,
  ApiError,
  type Candidate,
  type CandidateAnalysis,
  type CandidateRefinement,
  type RefinementPriority
} from "@/lib/api-client";

function timestamp(value: number) {
  const minutes = Math.floor(value / 60);
  const seconds = Math.floor(value % 60);
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

const DISPOSITION_LABELS: Record<Candidate["disposition"], string> = {
  CANDIDATE: "Candidate",
  CANDIDATE_NEEDS_REFINEMENT: "Needs refinement",
  DO_NOT_CLIP: "Do not clip",
  DO_NOT_CLIP_RECENTLY_REDUNDANT: "Redundant"
};

function summaryLine(analysis: CandidateAnalysis): string {
  const metrics = analysis.metrics ?? {};
  const retained = Number(metrics.candidates_retained ?? 0);
  const refinement = Number(metrics.refinement_needed_count ?? 0);
  const redundant = Number(metrics.redundant_count ?? 0);
  const parts = [
    analysis.semantic_provider_mode,
    analysis.provider_status,
    analysis.cache_eligible ? "cache eligible" : "not cache eligible",
    `${retained} retained`
  ];
  if (refinement > 0) parts.push(`${refinement} need refinement`);
  if (redundant > 0) parts.push(`${redundant} redundant`);
  return parts.join(" · ");
}

function refinementLabel(status: string) {
  return status.replaceAll("_", " ");
}

function recoveredTerms(refinement: CandidateRefinement) {
  const recovered = refinement.code_switch_evidence.code_switch_recovered;
  return Array.isArray(recovered) ? recovered.filter((term): term is string => typeof term === "string") : [];
}

function ManualRefinementReview({
  refinement,
  onSubmit
}: {
  refinement: CandidateRefinement;
  onSubmit?: (refinementId: string, text: string, resolutions: Record<string, string>) => void;
}) {
  const [text, setText] = useState(refinement.manual_transcript ?? refinement.final_transcript);
  const [resolutions, setResolutions] = useState<Record<string, string>>({});
  return (
    <details className="candidate-manual-review" open>
      <summary>Manual transcript review</summary>
      {refinement.unresolved_spans.map((span, index) => {
        const spanId = typeof span.span_id === "string" ? span.span_id : `span-${index}`;
        const readings = Array.isArray(span.readings) ? span.readings.filter((item): item is string => typeof item === "string") : [];
        const reason = typeof span.reason === "string" ? span.reason.replaceAll("_", " ") : "unresolved evidence";
        return (
          <div key={spanId}>
            <p><strong>{reason}</strong>{readings.length ? `: ${readings.join(", ")}` : ""}</p>
            <input
              aria-label={`Resolution for ${spanId}`}
              onChange={(event) => setResolutions((current) => ({ ...current, [spanId]: event.target.value }))}
              placeholder="Explicit resolution"
              value={resolutions[spanId] ?? ""}
            />
          </div>
        );
      })}
      <textarea aria-label="Manual final transcript" dir="auto" onChange={(event) => setText(event.target.value)} value={text} />
      <button
        className="button"
        disabled={!onSubmit || !text.trim()}
        onClick={() => onSubmit?.(refinement.id, text, resolutions)}
        type="button"
      >
        Save manual transcript
      </button>
    </details>
  );
}

function RefinementResult({
  refinement,
  onSeek,
  onManualSubmit
}: {
  refinement: CandidateRefinement;
  onSeek: (seconds: number) => void;
  onManualSubmit?: (refinementId: string, text: string, resolutions: Record<string, string>) => void;
}) {
  const start = refinement.refined_start ?? refinement.coarse_start;
  const end = refinement.refined_end ?? refinement.coarse_end;
  const recovered = recoveredTerms(refinement);
  return (
    <div className="candidate-refinement" dir="auto">
      <p>
        <strong>{refinement.priority === "FINAL_CLIP" ? "Final-clip refinement" : "Candidate refinement"}</strong>
        {` · ${refinementLabel(refinement.status)} · ${Math.round(refinement.confidence * 100)}% confidence`}
      </p>
      <button className="button" onClick={() => onSeek(start)} type="button">
        Refined window {timestamp(start)}–{timestamp(end)}
      </button>
      <p><MixedDirectionText text={refinement.final_transcript || refinement.automatic_transcript || "Transcript is still being prepared."} /></p>
      <p className="muted">
        Coarse {timestamp(refinement.coarse_start)}–{timestamp(refinement.coarse_end)}
        {refinement.dialect_profile ? ` · ${refinement.dialect_profile}` : ""}
        {recovered.length ? ` · recovered: ${recovered.join(", ")}` : ""}
        {refinement.unresolved_spans.length ? ` · ${refinement.unresolved_spans.length} unresolved` : ""}
      </p>
      {refinement.priority === "FINAL_CLIP" && refinement.status === "NEEDS_MANUAL_TRANSCRIPT_REVIEW" && (
        <ManualRefinementReview refinement={refinement} onSubmit={onManualSubmit} />
      )}
    </div>
  );
}

export function CandidateList({
  analysis,
  candidates,
  refinements,
  onSeek,
  onQueueRefinement,
  onManualRefinement,
  actions
}: {
  analysis: CandidateAnalysis | null;
  candidates: Candidate[];
  refinements?: CandidateRefinement[];
  onSeek: (seconds: number) => void;
  onQueueRefinement?: (candidateId: string, priority: RefinementPriority) => void;
  onManualRefinement?: (refinementId: string, text: string, resolutions: Record<string, string>) => void;
  actions?: React.ReactNode;
}) {
  if (analysis === null) {
    return (
      <section className="card" dir="auto">
        <h3>Clip candidates</h3>
        <p className="muted">No candidate analysis yet.</p>
      </section>
    );
  }
  return (
    <section className="card" dir="auto">
      <h3>Clip candidates</h3>
      <p className="muted">{summaryLine(analysis)}</p>
      {actions}
      {candidates.length === 0 ? (
        <p className="muted">No candidates.</p>
      ) : (
        <div className="transcript-segments">
          {candidates.map((candidate) => {
            const candidateRefinements = (refinements ?? []).filter((item) => item.clip_candidate_id === candidate.id);
            const refinable = candidate.disposition === "CANDIDATE" || candidate.disposition === "CANDIDATE_NEEDS_REFINEMENT";
            return (
            <div className="transcript-segment" key={candidate.id}>
              <button onClick={() => onSeek(candidate.start_time)} type="button">
                <time>
                  {timestamp(candidate.start_time)}–{timestamp(candidate.end_time)}
                </time>
                <MixedDirectionText text={candidate.transcript_excerpt} />
              </button>
              <p className="muted">
                <span className={candidate.disposition === "CANDIDATE" ? "ok" : undefined}>
                  {DISPOSITION_LABELS[candidate.disposition]}
                </span>
                {` · ${candidate.primary_content_type.replace(/_/g, " ")}`}
                {` · score ${Math.round(candidate.clip_score * 100)}%`}
                {candidate.dialect_profile ? ` · ${candidate.dialect_profile}` : ""}
                {candidate.code_switch_suspected ? " · code-switch" : ""}
              </p>
              {candidate.refinement_reasons.length > 0 && (
                <p className="muted">Needs refinement: {candidate.refinement_reasons.join(", ")}</p>
              )}
              {refinable && (
                <div>
                  <button className="button" onClick={() => onQueueRefinement?.(candidate.id, "CANDIDATE")} type="button">
                    Refine candidate
                  </button>
                  <button className="button" onClick={() => onQueueRefinement?.(candidate.id, "FINAL_CLIP")} type="button">
                    Refine as final clip
                  </button>
                </div>
              )}
              {candidateRefinements.map((refinement) => (
                <RefinementResult key={refinement.id} refinement={refinement} onManualSubmit={onManualRefinement} onSeek={onSeek} />
              ))}
            </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

export function CandidateResults({
  sourceId,
  revision,
  onSeek,
  onRefinementQueued
}: {
  sourceId: string;
  revision: number;
  onSeek: (seconds: number) => void;
  onRefinementQueued?: () => void;
}) {
  const [includeRejected, setIncludeRejected] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [queueError, setQueueError] = useState("");
  const load = useCallback(
    () =>
      Promise.all([
        api.getCandidateAnalysis(sourceId).catch((cause) => {
          if (cause instanceof ApiError && cause.status === 404) return null;
          throw cause;
        }),
        api.listCandidates(sourceId, { includeRejected })
      ]).then(async ([analysis, candidates]) => ({
        analysis,
        candidates,
        refinements: (await Promise.all(candidates.map((candidate) => api.listCandidateRefinements(candidate.id)))).flat()
      })),
    [sourceId, includeRejected]
  );
  const queueRefinement = async (candidateId: string, priority: RefinementPriority) => {
    setQueueError("");
    try {
      await api.queueCandidateRefinement(candidateId, priority);
      setRefresh((value) => value + 1);
      onRefinementQueued?.();
    } catch (cause) {
      setQueueError(cause instanceof Error ? cause.message : "Could not queue refinement");
    }
  };
  const submitManualRefinement = async (
    refinementId: string,
    text: string,
    resolutions: Record<string, string>
  ) => {
    setQueueError("");
    try {
      await api.submitManualCandidateTranscript(refinementId, text, resolutions);
      setRefresh((value) => value + 1);
    } catch (cause) {
      setQueueError(cause instanceof Error ? cause.message : "Could not save manual transcript");
    }
  };
  return (
    <ApiState key={`${revision}-${refresh}`} load={load}>
      {({ analysis, candidates, refinements }) => (
        <>
          <CandidateList
            actions={
              <button
                className="button"
                onClick={() => setIncludeRejected((value) => !value)}
                type="button"
              >
                {includeRejected ? "Hide rejected proposals" : "Show rejected proposals"}
              </button>
            }
            analysis={analysis}
            candidates={candidates}
            onManualRefinement={(refinementId, text, resolutions) => void submitManualRefinement(refinementId, text, resolutions)}
            onSeek={onSeek}
            onQueueRefinement={(candidateId, priority) => void queueRefinement(candidateId, priority)}
            refinements={refinements}
          />
          {queueError && <p className="error">Could not queue refinement: {queueError}</p>}
        </>
      )}
    </ApiState>
  );
}
