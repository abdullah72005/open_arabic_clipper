"use client";

import { ProduceRunner } from "@/components/produce-runner";

export default function ProducePage() {
  return (
    <>
      <h2>Produce final clip</h2>
      <p className="muted">
        Paste a public, permitted media link. The app ingests and transcribes it, picks the
        highest-scoring retained clip candidate, then runs transcript refinement, transformation
        analysis, planning, governance, selection, composition, and render — and plays the final MP4.
      </p>
      <ProduceRunner />
    </>
  );
}
