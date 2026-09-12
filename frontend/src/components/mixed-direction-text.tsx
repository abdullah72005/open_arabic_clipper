import React from "react";

interface MixedDirectionTextProps {
  text: string;
  className?: string;
}

/** Renders logical Unicode text in an isolated, automatically resolved BiDi scope. */
export function MixedDirectionText({ text, className }: MixedDirectionTextProps) {
  return <bdi className={className} dir="auto">{text}</bdi>;
}
