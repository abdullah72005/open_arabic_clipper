"use client";

import React, { useCallback, useState } from "react";

import { ApiState } from "@/components/api-state";
import { api, ApiError, type Candidate, type CandidateAnalysis } from "@/lib/api-client";

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

export function CandidateList({
  analysis,
  candidates,
  onSeek,
  actions
}: {
  analysis: CandidateAnalysis | null;
  candidates: Candidate[];
  onSeek: (seconds: number) => void;
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
          {candidates.map((candidate) => (
            <div className="transcript-segment" key={candidate.id}>
              <button onClick={() => onSeek(candidate.start_time)} type="button">
                <time>
                  {timestamp(candidate.start_time)}–{timestamp(candidate.end_time)}
                </time>
                <span>{candidate.transcript_excerpt}</span>
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
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

export function CandidateResults({
  sourceId,
  revision,
  onSeek
}: {
  sourceId: string;
  revision: number;
  onSeek: (seconds: number) => void;
}) {
  const [includeRejected, setIncludeRejected] = useState(false);
  const load = useCallback(
    () =>
      Promise.all([
        api.getCandidateAnalysis(sourceId).catch((cause) => {
          if (cause instanceof ApiError && cause.status === 404) return null;
          throw cause;
        }),
        api.listCandidates(sourceId, { includeRejected })
      ]).then(([analysis, candidates]) => ({ analysis, candidates })),
    [sourceId, includeRejected]
  );
  return (
    <ApiState key={revision} load={load}>
      {({ analysis, candidates }) => (
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
          onSeek={onSeek}
        />
      )}
    </ApiState>
  );
}
