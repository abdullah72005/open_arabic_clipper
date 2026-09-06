# Processing and Provenance Separation

## Decision

ClipFactory records source provenance and rights status independently from
pipeline processing.  A technically accessible public URL may enter local
ingest, transcription, correction, quality analysis, and review with an
unknown or third-party provenance state.  The recorded value is never
upgraded by processing.

## Boundaries

`IngestExecutor` decides only whether the supported acquisition mechanism can
obtain the source.  It keeps all existing URL validation and acquisition
failures.  A future publishing governor is the enforcement boundary for
rights, licence, commercial-use, copyright-risk, and originality review.

## Existing Sources and UI

Jobs that previously failed only because rights were unknown can be retried
without recreating the source.  Source details show the latest job, its error,
and a retry control for failed or cancelled work.
