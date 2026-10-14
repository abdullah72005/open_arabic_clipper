"use client";

import Link from "next/link";
import { useCallback } from "react";

import { ApiState } from "@/components/api-state";
import { api } from "@/lib/api-client";

export default function ResultsPage() {
  const load = useCallback(() => api.listRenderArtifacts(100), []);
  return (
    <>
      <h2>Results</h2>
      <p className="muted">Rendered final clips (Stage 5.2). Every finished render appears here.</p>
      <ApiState load={load}>
        {(artifacts) =>
          artifacts.length ? (
            <div className="grid">
              {artifacts.map((artifact) => {
                const url = api.renderExecutionArtifactUrl(artifact.id);
                return (
                  <section className="card" key={artifact.id}>
                    <video
                      controls
                      playsInline
                      preload="metadata"
                      src={url}
                      style={{ width: "100%", borderRadius: 12 }}
                    />
                    <p className="muted">
                      {new Date(artifact.created_at).toLocaleString()} ·{" "}
                      {artifact.qc_status ?? artifact.lifecycle}
                    </p>
                    <p>
                      <a className="button" download href={url}>
                        Download MP4
                      </a>{" "}
                      <Link className="button" href={`/sources/${artifact.source_video_id}`}>
                        Source
                      </Link>
                    </p>
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
