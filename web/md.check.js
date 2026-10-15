// Checks for the escape-first renderer in md.js. Run: node web/md.check.js
"use strict";
global.window = {};
require("./md.js");
const render = window.HonneMD.render;

// [input, what the output must contain or must not contain]
const cases = [
  ["# Hi **there**", { has: "<h3>Hi <strong>there</strong></h3>" }],
  ["<img src=x onerror=alert(1)>", { has: "&lt;img", not: "<img" }],
  ["```\n<script>x</script>\n```", { has: "<pre><code>&lt;script&gt;", not: "<script>" }],
  ["`<b>` and *em* and snake_case_word", { has: "<code>&lt;b&gt;</code> and <em>em</em> and snake_case_word" }],
  ["- a\n- **b**\n\n1. one", { has: "<ul><li>a</li><li><strong>b</strong></li></ul>" }],
  ["[ok](https://a.com/x)", { has: '<a href="https://a.com/x" target="_blank"' }],
  ["[q](https://a.com/?b=\"><script>)", { not: "<script>" }],
  ["[d](/datasets/circuits/ioi_public.json)", { has: '<a href="/datasets/circuits/ioi_public.json">d</a>' }],
  ["[p](/a/b?x=1&y=2#h)", { has: '<a href="/a/b?x=1&amp;y=2#h">' }],
  ["[x](javascript:alert(1))", { not: "<a " }],
  ["[x](data:text/html,hi)", { not: "<a " }],
  ["[x](vbscript:x)", { not: "<a " }],
  ["[x](//evil.com/a)", { not: "<a " }],
  ["[x](/\\evil.com)", { not: "<a " }],
  ["[x](/a//evil.com)", { not: "<a " }],
  ["[x](/)", { not: "<a " }],
  ["[x](/a\"onmouseover=\"alert(1))", { not: "<a " }],
];

let failed = 0;
for (const [src, want] of cases) {
  const out = render(src);
  const ok = (!want.has || out.includes(want.has)) && (!want.not || !out.includes(want.not));
  if (!ok) { failed++; console.log("FAIL", JSON.stringify(src), "=>", out); }
}
console.log(failed ? failed + " of " + cases.length + " failed" : "all " + cases.length + " passed");
process.exit(failed ? 1 : 0);
