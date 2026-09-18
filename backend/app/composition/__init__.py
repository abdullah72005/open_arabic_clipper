"""Stage 5.1 vertical visual-composition, framing, and caption rendering plan.

This package produces a deterministic, CPU-local, provider-free *plan* for a
selected clip candidate. It is not a rendered video.

What it does:
- consumes only the executable, current Stage 5.0 render contract;
- analyzes only the selected bound source spans (plus a bounded scene-context
  margin used solely for cut alignment);
- runs a vendored YuNet ONNX face detector on the local CPU through the already
  installed ``onnxruntime`` (no new libraries, no OpenCV, no MediaPipe);
- builds anonymous per-scene tracks, a deterministic per-scene framing mode, a
  compact crop-keyframe path, FINAL_CLIP-only caption events, one ASS document,
  and materialization-required overlay requirements;
- persists a ``visual_composition_plans`` row and exposes a read-only Stage 5.2
  handoff.

What it never does:
- no final production FFmpeg render, no encoding (libx264/aac/loudnorm), no final
  MP4 persistence, no final audio mix, no render QC, no black-frame analysis of a
  production render;
- no TTS/voice/model selection, no narration generation, no hook writing, no
  B-roll/gameplay/image generation, no publishing/scheduling/metadata;
- no Gemini, Qwen/Ollama, hosted vision API, or any network I/O at analysis time;
- no whole-source visual scanning, no active-speaker/AV ML, no split-screen, no
  face recognition/identity/biometrics, no platform evasion;
- no Stage 4 or Stage 5.0 mutation and no Stage 6 materialization.
"""

__all__: list[str] = []
