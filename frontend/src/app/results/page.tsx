"use client";

import Link from "next/link";
import { useCallback } from "react";

import { ApiState } from "@/components/api-state";
import { api, type RenderArtifact } from "@/lib/api-client";

const STATE_LABELS: Record<RenderArtifact["materialization_state"], string> = {
  CORE: "Core source validation (Stage 5.2)",
  REVIEW_READY: "Review-ready transformed short",
  PENDING: "Pending",
  DEFERRED: "Deferred",
  REJECTED: "Rejected",
  NO_OUTPUT: "No output (valid)",
  BLOCKED: "Blocked",
  FAILED: "Failed",
  CANCELLED: "Cancelled",
  HISTORICAL: "Historical (not review-ready)",
  DIAGNOSTIC: "Diagnostic media"
};

function artifactUrl(artifact: RenderArtifact): string {
  if (artifact.artifact_endpoint === "materialization" && artifact.materialization_id) {
    return api.materializationArtifactUrl(artifact.materialization_id);
  }
  return api.renderExecutionArtifactUrl(artifact.id);
}

export default function ResultsPage() {
  const load = useCallback(() => api.listRenderArtifacts(100), []);
  return (
    <>
      <h2>Results</h2>
      <p className="muted">
        Rendered media. <strong>Transformed Shorts</strong> (Stage 6) are reviewable only when the
        committed assessment permits review; core source validations (Stage 5.2) are technical
        checks, not final clips. Historical or diagnostic media remains inspectable but is never
        current review readiness.
      </p>
      <ApiState load={load}>
        {(artifacts) =>
          artifacts.length ? (
            <div className="grid">
              {artifacts.map((artifact) => {
                const url = artifactUrl(artifact);
                const transformed = artifact.kind === "TRANSFORMED_SHORT";
                const reviewReady = transformed && artifact.review_ready;
                return (
                  <section className="card" key={artifact.id}>
                    <video
                      controls
                      playsInline
                      preload="metadata"
                      src={url}
                      style={{ width: "100%", borderRadius: 12 }}
                    />
                    <p>
                      <strong>
                        {transformed ? "Transformed Short (Stage 6)" : "Core source validation (Stage 5.2)"}
                      </strong>
                      {reviewReady && <span className="badge"> review-ready</span>}
                    </p>
                    {transformed && (
                      <p className="muted">
                        {STATE_LABELS[artifact.materialization_state]}
                        {artifact.semantic_outcome ? ` · ${artifact.semantic_outcome}` : ""}
                        {` · lifecycle ${artifact.materialization_lifecycle ?? artifact.lifecycle}`}
                      </p>
                    )}
                    {transformed && !reviewReady && (
                      <p className="muted">
                        Not a current review-ready clip
                        {artifact.readiness_reasons.length
                          ? ` (${artifact.readiness_reasons.slice(0, 4).join(", ")})`
                          : ""}
                        .
                      </p>
                    )}
                    {transformed && artifact.findings.length > 0 && (
                      <ul className="muted">
                        {artifact.findings.slice(0, 6).map((finding, index) => (
                          <li key={index}>
                            {String(finding.dimension ?? "finding")}: {String(finding.level ?? "")}
                          </li>
                        ))}
                      </ul>
                    )}
                    <p className="muted">
                      {new Date(artifact.created_at).toLocaleString()} ·{" "}
                      {artifact.qc_status ?? artifact.lifecycle}
                      {artifact.materialization_id
                        ? ` · materialization ${artifact.materialization_id.slice(0, 8)}`
                        : ""}
                    </p>
                    <p>
                      <a className="button" download href={url}>
                        Download MP4
                      </a>{" "}
                      <Link className="button" href={`/sources/${artifact.source_video_id}`}>
                        Source
                      </Link>
                    </p>
                    <p className="muted">Publication eligibility is evaluated separately.</p>
                  </section>
                );
              })}
            </div>
          ) : (
            <p className="muted">
              No rendered clips yet. Use <Link href="/produce">Produce</Link> to generate one.
            </p>
          )
        }
      </ApiState>
    </>
  );
}
