"use client";
import { useEffect, useState } from "react";

export function ApiState<T>({ load, children }: { load: () => Promise<T>; children: (value: T) => React.ReactNode }) {
  const [state, setState] = useState<{ value?: T; error?: string }>({});
  useEffect(() => { load().then((value) => setState({ value })).catch((error: Error) => setState({ error: error.message })); }, [load]);
  if (state.error) return <p className="error">Could not load data: {state.error}</p>;
  if (!state.value) return <p className="muted">Loading…</p>;
  return <>{children(state.value)}</>;
}
