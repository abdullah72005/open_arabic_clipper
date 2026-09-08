"use client";

import { FormEvent, useState } from "react";

import { ApiError, api, RightsStatus } from "@/lib/api-client";

const rightsOptions: Array<{ value: RightsStatus; label: string }> = [
  { value: "UNKNOWN", label: "Unknown — process locally; publishing requires review" },
  { value: "THIRD_PARTY_UNKNOWN", label: "Third-party — rights unknown" },
  { value: "THIRD_PARTY_REUSE", label: "Third-party — reuse claimed" },
  { value: "OWNED", label: "Owned" },
  { value: "LICENSED", label: "Licensed" },
  { value: "PERMISSION", label: "Permission granted" },
  { value: "PUBLIC_DOMAIN", label: "Public domain" },
  { value: "OTHER_ALLOWED", label: "Other allowed use" }
];

export default function AddSourcePage() {
  const [urls, setUrls] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [rightsStatus, setRightsStatus] = useState<RightsStatus>("UNKNOWN");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const entries = urls
      .split("\n")
      .map((url) => url.trim())
      .filter(Boolean);
    if (!entries.length && !file) {
      setMessage("Enter one or more URLs, or select a file.");
      return;
    }

    setBusy(true);
    setMessage("");
    try {
      if (entries.length) await api.submitUrls(entries, rightsStatus);
      if (file) await api.upload(file, rightsStatus);
      setUrls("");
      setFile(null);
      setMessage("Submitted for ingest and probing.");
    } catch (error) {
      setMessage(error instanceof ApiError ? error.message : "Submission failed.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <h2>Add source</h2>
      <p className="muted">
        Only submit media you own or have permission to process. URL sources must be public and
        permitted; this app never bypasses platform protections.
      </p>
      <form className="card" onSubmit={submit}>
        <label>
          Rights status
          <select
            value={rightsStatus}
            onChange={(event) => setRightsStatus(event.target.value as RightsStatus)}
          >
            {rightsOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          One URL per line
          <textarea
            rows={8}
            value={urls}
            onChange={(event) => setUrls(event.target.value)}
            placeholder="https://example.org/video"
          />
        </label>
        <p className="muted">or</p>
        <label>
          Local media file
          <input
            type="file"
            accept="video/*,audio/*"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          />
        </label>
        <p>
          <button className="button" disabled={busy}>
            {busy ? "Submitting…" : "Queue source"}
          </button>
        </p>
        {message && <p className={message.startsWith("Submitted") ? "ok" : "error"}>{message}</p>}
      </form>
    </>
  );
}
