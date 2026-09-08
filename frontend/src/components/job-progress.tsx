"use client";
import React from "react";
import { useEffect, useState } from "react";
import type { Job } from "@/lib/api-client";

export const KIND_LABELS: Record<string, string> = {
  INGEST: "Ingesting media",
  PROBE: "Probing media",
  TRANSCRIPTION: "Transcribing audio",
  RECONSTRUCTION: "Reconstructing transcript"
};

export function formatElapsed(startedAt: string | null, now: number): string {
  if (!startedAt) return "waiting to start";
  const seconds = Math.max(0, Math.floor((now - new Date(startedAt).getTime()) / 1000));
  const minutes = Math.floor(seconds / 60);
  return minutes > 0 ? `${minutes}m ${seconds % 60}s` : `${seconds}s`;
}

export function JobProgress({ job }: { job: Job }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  const label = KIND_LABELS[job.kind] ?? job.kind;
  const statusLabel =
    job.status === "RUNNING" ? "running" : job.status === "QUEUED" ? "queued" : job.status;
  return (
    <section aria-live="polite" className="card">
      <h3>Processing</h3>
      <p>
        <strong>{label}</strong> — {statusLabel}
      </p>
      <p className="muted">Elapsed: {formatElapsed(job.started_at, now)}</p>
      {job.status === "QUEUED" && (
        <p className="muted">Waiting for a worker to pick the job up.</p>
      )}
    </section>
  );
}