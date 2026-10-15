// Honne Arena: a tiny escape-first markdown renderer.
//
// Every server string is untrusted. This renderer HTML-escapes the whole input first,
// then adds back a fixed set of tags it builds itself: headings, paragraphs, lists,
// fenced code, inline code, bold, italic, and http(s) links. Nothing else. No raw HTML,
// no images, no attributes from the input except a link href that has already been
// escaped and must start with http:// or https://.
//
// This is the ONLY place the app produces HTML from server data (via x-html). Keep it
// small enough to read in one sitting.
//
// Links: http(s) URLs open in a new tab. Same-site absolute paths such as
// /datasets/x.json are allowed too: one "/" then a non-slash, with no "//" and no
// backslash anywhere, so a protocol-relative //host can never slip through.

(function (global) {
  "use strict";

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // Emphasis on text that is ALREADY escaped. Escaped text has no < > " ' left, so
  // the tags added here cannot be broken out of.
  function emphasis(s) {
    return s
      .replace(/\*\*(?=\S)([^*]*?\S)\*\*/g, "<strong>$1</strong>")
      .replace(/__(?=\S)([^_]*?\S)__/g, "<strong>$1</strong>")
      .replace(/(^|[^*\w])\*(?=\S)([^*]*?\S)\*(?![*\w])/g, "$1<em>$2</em>")
      .replace(/(^|[^_\w])_(?=\S)([^_]*?\S)_(?![_\w])/g, "$1<em>$2</em>");
  }

  // Links on escaped text. The URL group cannot contain a quote or angle bracket
  // (they were escaped to entities, and the classes below stop at whitespace and ")").
  var LINK = /\[([^\]\n]+)\]\((https?:\/\/[^\s()]+|\/[^\/\\\s()][^\s()\\]*)\)/g;

  function isSitePath(url) {
    return url.charAt(0) === "/" && url.charAt(1) !== "/" && url.indexOf("//") === -1 &&
      url.indexOf("\\") === -1;
  }

  function inlineText(raw) {
    var esc = escapeHtml(raw);
    var links = [];
    esc = esc.replace(LINK, function (m, text, url) {
      var site = url.charAt(0) === "/";
      if (site && !isSitePath(url)) return m;
      links.push(site
        ? '<a href="' + url + '">' + emphasis(text) + "</a>"
        : '<a href="' + url + '" target="_blank" rel="noopener noreferrer nofollow">' + emphasis(text) + "</a>");
      return "\u0000" + (links.length - 1) + "\u0000";
    });
    esc = emphasis(esc);
    return esc.replace(/\u0000(\d+)\u0000/g, function (_m, i) { return links[+i]; });
  }

  // Inline code spans first, so nothing inside backticks is treated as markup.
  function inline(raw) {
    raw = String(raw).replace(/\u0000/g, "");
    var parts = raw.split(/(`[^`\n]+`)/);
    var out = "";
    for (var i = 0; i < parts.length; i++) {
      var p = parts[i];
      if (p.length > 2 && p[0] === "`" && p[p.length - 1] === "`") {
        out += "<code>" + escapeHtml(p.slice(1, -1)) + "</code>";
      } else {
        out += inlineText(p);
      }
    }
    return out;
  }

  var FENCE = /^\s{0,3}(```|~~~)/;
  var HEADING = /^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
  var UL = /^\s{0,3}[-*+]\s+(.*)$/;
  var OL = /^\s{0,3}\d{1,9}[.)]\s+(.*)$/;

  function isBlockStart(line) {
    return FENCE.test(line) || HEADING.test(line) || UL.test(line) || OL.test(line);
  }

  // `headingBase` shifts levels so a description's "#" never outranks the page's own
  // headings. With base 3, "#" becomes <h3>.
  function render(src, opts) {
    var base = (opts && opts.headingBase) || 3;
    var lines = String(src == null ? "" : src).replace(/\r\n?/g, "\n").split("\n");
    var html = [];
    var i = 0;
    while (i < lines.length) {
      var line = lines[i];

      if (!line.trim()) { i++; continue; }

      var fence = line.match(FENCE);
      if (fence) {
        var marker = fence[1];
        var code = [];
        i++;
        while (i < lines.length && lines[i].trim().indexOf(marker) !== 0) {
          code.push(lines[i]);
          i++;
        }
        i++; // closing fence (or end of input)
        html.push("<pre><code>" + escapeHtml(code.join("\n")) + "</code></pre>");
        continue;
      }

      var h = line.match(HEADING);
      if (h) {
        var level = Math.min(6, base - 1 + h[1].length);
        html.push("<h" + level + ">" + inline(h[2]) + "</h" + level + ">");
        i++;
        continue;
      }

      var ordered = OL.test(line);
      if (ordered || UL.test(line)) {
        var re = ordered ? OL : UL;
        var items = [];
        while (i < lines.length) {
          var m = lines[i].match(re);
          if (m) {
            items.push(m[1]);
          } else if (lines[i].trim() && /^\s{2,}/.test(lines[i]) && items.length) {
            items[items.length - 1] += " " + lines[i].trim(); // continuation line
          } else {
            break;
          }
          i++;
        }
        var tag = ordered ? "ol" : "ul";
        html.push("<" + tag + ">" + items.map(function (t) {
          return "<li>" + inline(t) + "</li>";
        }).join("") + "</" + tag + ">");
        continue;
      }

      var para = [];
      while (i < lines.length && lines[i].trim() && !isBlockStart(lines[i])) {
        para.push(lines[i].trim());
        i++;
      }
      html.push("<p>" + inline(para.join("\n")) + "</p>");
    }
    return html.join("\n");
  }

  global.HonneMD = { render: render, escapeHtml: escapeHtml };
})(window);
