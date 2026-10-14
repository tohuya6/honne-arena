// Magic-link landing page. It must NOT use the token on load: Outlook Safe Links and
// other scanners fetch every emailed link, and the token works only once. The token is
// sent only when a person presses "Sign in".
"use strict";

(function () {
  const RETURN_KEY = "honne_return";
  const form = document.getElementById("verify-form");
  const btn = document.getElementById("verify-btn");
  const intro = document.getElementById("verify-intro");
  const status = document.getElementById("verify-status");

  function readToken() {
    const q = new URLSearchParams(location.search);
    const fromHash = new URLSearchParams(location.hash.replace(/^#/, ""));
    return (q.get("token") || fromHash.get("token") || "").trim();
  }

  function say(text, tone) {
    status.textContent = "";
    const p = document.createElement("p");
    p.className = tone === "bad" ? "error-text" : "ok-text";
    p.textContent = text;
    status.appendChild(p);
  }

  function addLink(text, href) {
    const p = document.createElement("p");
    const a = document.createElement("a");
    a.href = href;
    a.textContent = text;
    p.appendChild(a);
    status.appendChild(p);
  }

  // Only same-site hash routes, so a stored value can never send you off-site.
  function returnTo(user) {
    let back = "#/";
    try { back = sessionStorage.getItem(RETURN_KEY) || "#/"; sessionStorage.removeItem(RETURN_KEY); } catch (_) { /* private mode */ }
    if (!/^#\/[\w\-./?=&%]*$/.test(back) || back.startsWith("#/signin")) back = "#/";
    if (user && !user.handle) back = "#/me";
    return "/" + back;
  }

  const token = readToken();
  if (!token) {
    intro.textContent = "This sign-in link is missing its code. Ask for a new link and open it straight from the email.";
    btn.hidden = true;
    addLink("Get a new link", "/#/signin");
  }

  form.addEventListener("submit", async function (ev) {
    ev.preventDefault();
    if (!token || btn.disabled) return;
    btn.disabled = true;
    btn.textContent = "Signing in...";
    let res, data = null;
    try {
      res = await fetch("/api/auth/verify", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ token: token }),
      });
      try { data = await res.json(); } catch (_) { data = null; }
    } catch (_) {
      say("Can't reach the server. Check your connection and try again.", "bad");
      btn.disabled = false;
      btn.textContent = "Sign in";
      return;
    }
    if (!res.ok) {
      const msg = data && typeof data.error === "string" && data.error
        ? data.error : "This link didn't work. It may have expired or been used already.";
      say(msg, "bad");
      addLink("Get a new link", "/#/signin");
      btn.hidden = true;
      return;
    }
    const user = data && data.user;
    say("You're signed in" + (user && user.handle ? " as " + user.handle : "") + ". Taking you to the arena...");
    // Drop the used token from the address bar and history.
    try { history.replaceState(null, "", "/verify.html"); } catch (_) { /* ignore */ }
    setTimeout(function () { location.replace(returnTo(user)); }, 900);
  });
})();
