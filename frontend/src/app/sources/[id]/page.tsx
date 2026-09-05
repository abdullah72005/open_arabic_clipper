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
  onSeek
}: {
  transcript: Transcript | null;
  onSeek: (seconds: number) => void;
}) {
  if (!transcript) return <p className="muted">Transcript is not ready yet.</p>;
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
          <button
            className="transcript-segment"
            key={`${segment.start}-${index}`}
            onClick={() => onSeek(segment.start)}
            type="button"
          >
            <time>{timestamp(segment.start)}</time>
            <span>{segment.normalized_text ?? segment.text}</span>
          </button>
        ))}
      </div>
    </section>
  );
}

export default function SourceDetail() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState("");
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
            <p className="muted">Transcription auto-detects Arabic, English, and mixed speech locally.</p>
            <button className="button danger" onClick={remove}>Delete source</button>
            {error && <p className="error">{error}</p>}
          </section>
          <ApiState load={loadTranscript}>
            {(transcript) => <TranscriptViewer onSeek={seekTo} transcript={transcript} />}
          </ApiState>
        </>
      )}
    </ApiState>
  );
}
