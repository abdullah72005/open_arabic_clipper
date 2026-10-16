import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("next/link", () => ({
  default: ({ children, href }: { children: React.ReactNode; href: string }) =>
    React.createElement("a", { href }, children)
}));

import { ProduceRunner } from "./produce-runner";

describe("ProduceRunner", () => {
  it("renders the produce form and the Stage 6 materialize step", () => {
    const markup = renderToStaticMarkup(<ProduceRunner />);
    expect(markup).toContain("Produce final clip");
    expect(markup).toContain("Materialize");
    expect(markup).toContain("Media link");
  });
});
