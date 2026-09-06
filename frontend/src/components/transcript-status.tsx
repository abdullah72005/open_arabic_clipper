import type { QualityResponse, ReconstructionStatus, Transcript } from "@/lib/api-client";

const statusLabels: Record<ReconstructionStatus, string> = {
  NOT_REQUIRED: "Reconstruction not required",
  APPLIED: "Reconstruction applied",
  UNCHANGED_HIGH_CONFIDENCE: "High-confidence transcript unchanged",
  LOW_CONFIDENCE_UNRESOLVED: "Low-confidence transcript unresolved",
  PROVIDER_UNAVAILABLE: "Reconstruction unavailable",
  FAILED: "Reconstruction failed",
  MANUAL_OVERRIDE: "Manual override applied"
};

function statusLabel(status: string) {
  return statusLabels[status as ReconstructionStatus] ?? status;
}

function percentage(value: number | null | undefined) {
  return value == null ? "Not available" : `${Math.round(value * 100)}%`;
}

function metadataString(metadata: Record<string, unknown>, key: string) {
  const value = metadata[key];
  return typeof value === "string" && value ? value : null;
}

export function TranscriptStatus({
  transcript,
  quality
}: {
  transcript: Transcript;
  quality: QualityResponse | null;
}) {
  const status = quality?.reconstruction_status ?? transcript.reconstruction_status;
  const metrics = quality?.quality;
  const metadata = transcript.reconstruction_metadata ?? {};
  const health = metadata.provider_health;
  const providerHealth = health && typeof health === "object"
    ? health as Record<string, unknown>
    : {};
  const availability = metadataString(providerHealth, "availability")
    ?? metadataString(metadata, "provider_availability")
    ?? "UNKNOWN";
  const model = metadataString(providerHealth, "model")
    ?? metadataString(metadata, "model")
    ?? transcript.reconstruction_method
    ?? "Not available";
  const focusSpans = transcript.segments.flatMap((segment) => segment.focus_spans ?? []);
  const routingReasons = Array.from(new Set(
    transcript.segments.flatMap((segment) => segment.routing_reasons ?? [])
  ));

  return (
    <div className="transcript-status">
      <p><strong className="status-badge">{statusLabel(status)}</strong></p>
      <dl>
        <dt>Provider</dt><dd>{availability}</dd>
        <dt>Model</dt><dd>{model}</dd>
        <dt>Audio quality</dt><dd>{percentage(metrics?.audio_quality_score)}</dd>
        <dt>Transcript quality</dt><dd>{percentage(metrics?.transcript_quality_score)}</dd>
        <dt>Low-confidence words</dt><dd>{percentage(metrics?.low_confidence_word_ratio)}</dd>
        <dt>Unresolved segments</dt><dd>{percentage(metrics?.unresolved_segment_ratio)}</dd>
        <dt>Conservative source floor</dt><dd>{percentage(metrics?.conservative_source_floor)}</dd>
      </dl>
      <p>{metrics?.manual_review_required ? "Manual review required" : "Manual review not required"}</p>
      {routingReasons.length > 0 && (
        <p><strong>Routing reasons:</strong> {routingReasons.join(", ")}</p>
      )}
      {focusSpans.length > 0 && (
        <p><strong>Focus spans:</strong> {focusSpans.map((span) => (
          <span key={`${span.word}-${span.start}-${span.end}`}> {span.word} · {percentage(span.probability)}</span>
        ))}</p>
      )}
    </div>
  );
}
