import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { MixedDirectionText } from "./mixed-direction-text";

const samples = [
  "يعني ممكن يكون عادي بس content creator دي مش شغلانة بالنسبة للبيت",
  "أنا عملت deploy للbackend امبارح",
  "هو بيستخدم Next.js مع Django",
  "السعر 500 جنيه",
  "السعر 500 USD تقريباً",
  "شوف https://example.com/docs وبعدها كمل",
  "أنا بستخدم React 19 دلوقتي",
  "النسخة الجديدة اسمها GPT-5.6 وطلعت امبارح",
  "ده endpoint اسمه /api/v2/users وبيشتغل عادي",
  'هو قال "content creator" مش موظف',
  "الـ backend (Django) شغال كويس",
  "C++ C# v2.1 /api/v2/users",
];

describe("MixedDirectionText", () => {
  it.each(samples)("keeps logical Unicode text intact while isolating %s", (text) => {
    const element = MixedDirectionText({ text });
    const markup = renderToStaticMarkup(<MixedDirectionText text={text} />);

    expect(element.props.children).toBe(text);
    expect(markup).toMatch(/^<bdi dir="auto">/u);
    expect(markup).toMatch(/<\/bdi>$/u);
    expect(markup).not.toMatch(/[\u2066-\u2069]/u);
  });
});
