"use client";

import { useParams, useRouter } from "next/navigation";
import { useCallback, useRef, useState } from "react";
import { ApiState } from "@/components/api-state";
import { api, ApiError, type Transcript } from "@/lib/api-client";

function timestamp(value: number) {
  const minutes = Math.floor(value / 60);
  const seconds = Math.floor(value % 60);
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

function TranscriptViewer({
  transcript,
  sourceId,
  onSeek
  ,
  onUpdated
}: {
  transcript: Transcript | null;
  sourceId: string;
  onSeek: (seconds: number) => void;
  onUpdated: () => void;
}) {
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [draft, setDraft] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  if (!transcript) return <p className="muted">Transcript is not ready yet.</p>;
  const startEditing = (index: number, text: string) => {
    setEditingIndex(index);
    setDraft(text);
    setError("");
  };
  const saveOverride = async (index: number) => {
    setSaving(true);
    setError("");
    try {
      await api.overrideTranscriptSegment(sourceId, index, draft);
      setEditingIndex(null);
      onUpdated();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not save transcript correction");
    } finally {
      setSaving(false);
    }
  };
  const clearOverride = async (index: number) => {
    setSaving(true);
    setError("");
    try {
      await api.clearTranscriptSegmentOverride(sourceId, index);
      onUpdated();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not clear transcript correction");
    } finally {
      setSaving(false);
    }
  };
  return (
    <section className="card transcript" dir="auto">
      <h3>Transcript</h3>
      <p className="muted">
        {transcript.language ?? "Auto-detected"}
        {transcript.detected_language_probability
          ? ` · ${Math.round(transcript.detected_language_probability * 100)}% confidence`
          : ""}
      </p>
      <div className="transcript-segments">
        {transcript.segments.map((segment, index) => (
          <div className="transcript-segment" key={`${segment.start}-${index}`}>
            <button onClick={() => onSeek(segment.start)} type="button">
              <time>{timestamp(segment.start)}</time>
              <span>{segment.final_text ?? segment.corrected_text ?? segment.normalized_text ?? segment.text}</span>
            </button>
            {(segment.correction_applied || segment.operator_text) && (
              <details>
                <summary>Correction details</summary>
                <p><strong>Raw:</strong> {segment.raw_text ?? segment.text}</p>
                <p><strong>Automatic:</strong> {segment.corrected_text ?? segment.normalized_text ?? segment.text}</p>
                <p className="muted">{segment.correction_method ?? "unchanged"} · {Math.round((segment.correction_confidence ?? 0) * 100)}%</p>
              </details>
            )}
            {editingIndex === index ? (
              <div>
                <textarea aria-label={`Transcript correction ${index + 1}`} value={draft} onChange={(event) => setDraft(event.target.value)} />
                <button className="button" disabled={saving || !draft.trim()} onClick={() => void saveOverride(index)} type="button">Save correction</button>
                <button className="button" disabled={saving} onClick={() => setEditingIndex(null)} type="button">Cancel</button>
              </div>
            ) : (
              <div>
                <button className="button" disabled={saving} onClick={() => startEditing(index, segment.final_text ?? segment.corrected_text ?? segment.text)} type="button">Edit correction</button>
                {segment.operator_text && <button className="button" disabled={saving} onClick={() => void clearOverride(index)} type="button">Clear manual text</button>}
              </div>
            )}
          </div>
        ))}
        {error && <p className="error">{error}</p>}
      </div>
    </section>
  );
}

export default function SourceDetail() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState("");
  const [transcriptRevision, setTranscriptRevision] = useState(0);
  const [jobRevision, setJobRevision] = useState(0);
  const load = useCallback(() => api.getSource(params.id), [params.id]);
  const loadTranscript = useCallback(
    () => api.getTranscript(params.id).catch((cause) => {
      if (cause instanceof ApiError && cause.status === 404) return null;
      throw cause;
    }),
    [params.id],
  );
  const seekTo = useCallback((seconds: number) => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = seconds;
    video.scrollIntoView({ behavior: "smooth", block: "center" });
    void video.play().catch(() => undefined);
  }, []);
  const remove = async () => {
    if (!confirm("Delete this source and its stored local files?")) return;
    try {
      await api.deleteSource(params.id);
      router.push("/sources");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Delete failed");
    }
  };
  const retry = async () => {
    setError("");
    try {
      await api.retrySource(params.id);
      setJobRevision((value) => value + 1);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Retry failed");
    }
  };

  return (
    <ApiState load={load}>
      {(source) => (
        <>
          <section className="card">
            <h2>{source.original_filename ?? "URL source"}</h2>
            {!source.source_uri.startsWith("http") && (
              <video
                controls
                id="source-preview"
                preload="metadata"
                ref={videoRef}
                src={api.sourceMediaUrl(source.id)}
              />
            )}
            <dl>
              <dt>Origin</dt><dd>{source.source_uri}</dd>
              <dt>Rights</dt><dd>{source.rights_status}</dd>
              <dt>Pipeline state</dt><dd>{source.lifecycle_state}</dd>
            </dl>
            <ApiState key={jobRevision} load={api.listJobs}>
              {(jobs) => {
                const job = jobs.find((item) => item.source_video_id === source.id);
                if (!job) return null;
                const canRetry = ["FAILED", "CANCELLED"].includes(job.status);
                return <div><strong>{job.kind} — {job.status}</strong>{job.error_message && <p className="error">Reason: {job.error_message}</p>}{canRetry && <button className="button" onClick={() => void retry()}>Retry</button>}</div>;
              }}
            </ApiState>
            <p className="muted">Transcription auto-detects Arabic, English, and mixed speech locally.</p>
            <button className="button danger" onClick={remove}>Delete source</button>
            {error && <p className="error">{error}</p>}
          </section>
          <ApiState key={transcriptRevision} load={loadTranscript}>
            {(transcript) => <TranscriptViewer onSeek={seekTo} onUpdated={() => setTranscriptRevision((value) => value + 1)} sourceId={source.id} transcript={transcript} />}
          </ApiState>
        </>
      )}
    </ApiState>
  );
}
