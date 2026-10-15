// Honne Arena frontend. Alpine.js over the JSON API in API.md.
//
// SECURITY: every server string is untrusted. Text goes into the page with x-text or
// textContent only. The single exception is markdown (challenge descriptions and hint
// bodies), which goes through HonneMD.render in md.js. That renderer escapes first and
// only adds tags it builds itself. Links from the server are bound with :href only after
// safeHref() accepts them.

"use strict";

const TRACKS = [
  { id: "ctf", name: "CTF" },
  { id: "lie-detection", name: "Lie detection" },
  { id: "steering", name: "Steering" },
  { id: "circuits", name: "Circuits" },
  { id: "cp", name: "Programming" },
];
const KINDS = ["flag", "predictions", "steering", "code", "circuits"];
const KIND_NAMES = {
  flag: "Flag", predictions: "Predictions", steering: "Steering",
  code: "Code", circuits: "Circuits",
};
const VERDICTS = {
  ac: "Accepted", accepted: "Accepted", ok: "Accepted",
  wa: "Wrong answer", wrong_answer: "Wrong answer",
  tle: "Time limit exceeded", time_limit: "Time limit exceeded", time_limit_exceeded: "Time limit exceeded",
  mle: "Memory limit exceeded", memory_limit: "Memory limit exceeded", memory_limit_exceeded: "Memory limit exceeded",
  re: "Runtime error", runtime_error: "Runtime error",
  ce: "Compile error", compile_error: "Compile error",
};
const HANDLE_RE = /^[\p{L}\p{N}_ \-.]{1,32}$/u;
const HEAD_RE = /^\d{1,3}\.\d{1,3}$/;
const MAX_UPLOAD_BYTES = 5 * 1024 * 1024;
const RETURN_KEY = "honne_return";
const SCAN_KEY = "honne_scanlines";

// Starter configs for the admin form. Placeholders only, never real secrets.
const CONFIG_TEMPLATES = {
  flag: { flag_format: "nuais{...}", case_sensitive: false },
  predictions: { metric: "auroc", labels_file: "data/private/<slug>.json",
                 public_item_ids_url: "/datasets/<slug>.csv" },
  steering: { model_id: "meta-llama/Llama-3.1-8B", layer: 16, d_file: "data/directions/<file>.npz",
              probe_file: "data/probes/<file>.json", token_budget: 20, higher_is_better: true },
  code: { language: "python", time_limit_s: 2, memory_mb: 256,
          samples: [{ input: "1 2\n", output: "3\n" }], tests_file: "data/private/<slug>/" },
  circuits: { max_heads: 10, task: "ioi" },
};

class ApiError extends Error {
  constructor(status, message, data) {
    super(message);
    this.status = status;
    this.data = data;
  }
}

const STATUS_TEXT = {
  400: "That didn't look right. Check it and try again.",
  401: "Please sign in first.",
  403: "You're not allowed to do that.",
  404: "We couldn't find that.",
  409: "That was already submitted.",
  429: "Too many tries. Wait a minute and try again.",
  503: "Scoring is offline right now. Try again later.",
  504: "The scorer took too long. Try again in a bit.",
};

async function api(method, path, body) {
  const opt = { method, headers: { Accept: "application/json" }, credentials: "same-origin" };
  if (body !== undefined) {
    opt.headers["Content-Type"] = "application/json";
    opt.body = JSON.stringify(body);
  }
  let r;
  try {
    r = await fetch(path, opt);
  } catch (_) {
    throw new ApiError(0, "Can't reach the server. Check your connection and try again.");
  }
  let data = null;
  try { data = await r.json(); } catch (_) { data = null; }
  if (!r.ok) {
    const msg = data && typeof data.error === "string" && data.error.trim()
      ? data.error : (STATUS_TEXT[r.status] || "Something went wrong (" + r.status + ").");
    throw new ApiError(r.status, msg, data);
  }
  return data;
}

function store(kind) {
  try { return kind === "session" ? window.sessionStorage : window.localStorage; } catch (_) { return null; }
}
function storeGet(kind, key) { try { return store(kind)?.getItem(key) ?? null; } catch (_) { return null; } }
function storeSet(kind, key, v) { try { store(kind)?.setItem(key, v); } catch (_) { /* private mode */ } }
function storeDel(kind, key) { try { store(kind)?.removeItem(key); } catch (_) { /* private mode */ } }

// Accept only http(s) URLs and same-site paths. Anything else (javascript:, data:, //host)
// becomes null and the link is not shown.
function safeHref(u) {
  if (typeof u !== "string" || !u.trim()) return null;
  const s = u.trim();
  // Same-site path: one "/" then a non-slash, and no "//" or backslash anywhere
  // (browsers read "/\\host" like "//host").
  if (s[0] === "/") return /^\/[^\/\\]/.test(s) && !s.includes("//") && !s.includes("\\") ? s : null;
  try {
    const url = new URL(s);
    return url.protocol === "https:" || url.protocol === "http:" ? url.href : null;
  } catch (_) {
    return null;
  }
}

function parseRoute(hash) {
  const h = (hash || "").replace(/^#/, "");
  if (!h.startsWith("/")) return null; // not ours (e.g. #main from the skip link)
  const [pathPart, query = ""] = h.split("?");
  const parts = pathPart.split("/").filter(Boolean).map((p) => {
    try { return decodeURIComponent(p); } catch (_) { return p; }
  });
  const q = Object.fromEntries(new URLSearchParams(query));
  if (!parts.length) return { name: "list", q };
  if (parts[0] === "c" && parts[1]) return { name: "detail", slug: parts[1], q };
  const known = ["leaderboard", "teams", "me", "signin", "admin"];
  if (known.includes(parts[0])) return { name: parts[0], q };
  return { name: "notfound", q };
}

function blankChallengeForm() {
  return {
    slug: "", title: "", kind: "flag", track: "ctf", tier: "learn", description: "",
    points_initial: 500, points_min: 100, points_decay: 10, hidden: true, closes_at: "",
    flag: "", configText: JSON.stringify(CONFIG_TEMPLATES.flag, null, 2), hints: [],
    // Kind-specific fields. Merged over the JSON config when set.
    k: { flag_format: "", case_sensitive: false, metric: "auroc", model_id: "", layer: "",
         token_budget: "", higher_is_better: true, language: "python", time_limit_s: "",
         memory_mb: "", max_heads: "" },
    preview: false,
  };
}

function arena() {
  return {
    // ── app state ──
    route: { name: "list", q: {} },
    season: null,
    me: { user: null, team: null, tiers: {} },
    meLoaded: false,
    tracks: TRACKS,
    kinds: KINDS,
    scanlines: false,

    challenges: [],
    listLoading: false,
    listError: "",
    filters: { track: "", tier: "" },

    // `detail` is never null once the view has rendered: Alpine re-evaluates child
    // bindings once before an x-if tears them down, so the view is gated on
    // `detailReady` instead and `detail` keeps its last value.
    detail: {},
    detailReady: false,
    detailLoading: false,
    detailError: "",
    answer: "",
    submitting: false,
    result: null,
    submitError: "",
    escArmed: false,
    fileNote: "",
    writeupBody: "",
    writeupMsg: "",
    writeupErr: "",
    hintErr: "",
    turnstileId: null,

    seasonBoard: null,
    boardSeason: "",
    boardError: "",
    allTime: [],
    allTimeTrack: "",
    allTimeError: "",

    handleInput: "",
    handleMsg: "",
    handleErr: "",
    teamName: "",
    joinCode: "",
    teamMsg: "",
    teamErr: "",
    createdCode: "",
    copied: false,

    email: "",
    signinSent: false,
    signinErr: "",
    devLink: null,
    busy: false,

    admin: {
      challenges: [], usage: null, error: "",
      season: { name: "", active: true, msg: "", err: "" },
      form: blankChallengeForm(), formMsg: "", formErr: "",
      hintFor: "", hintBody: "", hintCost: 0, hintMsg: "",
    },

    toasts: [],
    live: "off",
    dialog: { title: "", body: "", ok: "OK", resolve: null },

    _ws: null,
    _wsTries: 0,
    _wsTimer: null,
    _dirty: new Set(),
    _dirtyTimer: null,
    _toastId: 0,
    _titles: {},
    _titlesTried: false,

    // ── boot ──
    async init() {
      // Soham's CRT overlay is on by default. The footer toggle remembers "off".
      this.scanlines = storeGet("local", SCAN_KEY) !== "off";
      window.addEventListener("hashchange", () => this.onRoute());
      document.addEventListener("visibilitychange", () => {
        if (document.visibilityState === "visible" && this.live !== "on") this.connectLive(true);
      });
      await Promise.all([this.loadSeason(), this.loadMe()]);
      if (!location.hash || location.hash === "#") history.replaceState(null, "", "#/");
      await this.onRoute(true);
      this.connectLive();
    },

    async loadSeason() {
      try {
        this.season = await api("GET", "/api/season");
        if (this.season?.captcha_sitekey) this.loadTurnstile();
      } catch (_) {
        this.season = null;
      }
    },

    async loadMe() {
      try {
        const d = await api("GET", "/api/me");
        this.me = { user: d?.user || null, team: d?.team || null, tiers: d?.tiers || {} };
      } catch (_) {
        this.me = { user: null, team: null, tiers: {} };
      }
      this.meLoaded = true;
      this.handleInput = this.me.user?.handle || "";
    },

    get signedIn() { return !!this.me.user; },
    get needsHandle() { return !!this.me.user && !this.me.user.handle; },
    get isAdmin() { return !!this.me.user?.is_admin; },

    // ── routing ──
    async onRoute(first) {
      const r = parseRoute(location.hash);
      if (!r) return;
      this.route = r;
      document.title = this.pageTitle(r);
      if (r.name === "list") {
        this.filters.track = r.q.track || "";
        this.filters.tier = r.q.tier || "";
        await this.loadChallenges();
      } else if (r.name === "detail") {
        this.resetDetail();
        await this.loadDetail(r.slug);
      } else if (r.name === "leaderboard") {
        await Promise.all([this.loadBoard(), this.loadAllTime()]);
      } else if (r.name === "admin" && this.isAdmin) {
        await this.loadAdmin();
      } else if (r.name === "me" || r.name === "teams") {
        await this.loadMe();
      }
      if (!first) this.$nextTick(() => this.focusMain());
    },

    pageTitle(r) {
      const t = { list: "Challenges", leaderboard: "Leaderboard", teams: "Teams", me: "Me",
                  signin: "Sign in", admin: "Admin", notfound: "Not found" }[r.name];
      return (r.name === "detail" ? "Challenge" : t || "Honne Arena") + " | Honne Arena";
    },

    isRoute(...names) { return names.includes(this.route.name); },

    focusMain() {
      const h = document.querySelector("main h1");
      const target = h || document.getElementById("main");
      if (target) {
        if (!target.hasAttribute("tabindex")) target.setAttribute("tabindex", "-1");
        target.focus({ preventScroll: false });
        window.scrollTo({ top: 0 });
      }
    },

    goSignIn() {
      if (!this.isRoute("signin")) storeSet("session", RETURN_KEY, location.hash || "#/");
      location.hash = "#/signin";
    },

    // ── challenges list ──
    async loadChallenges() {
      this.listLoading = true;
      this.listError = "";
      const q = new URLSearchParams();
      if (this.filters.track) q.set("track", this.filters.track);
      if (this.filters.tier) q.set("tier", this.filters.tier);
      try {
        const d = await api("GET", "/api/challenges" + (q.toString() ? "?" + q : ""));
        const list = Array.isArray(d?.challenges) ? d.challenges : [];
        // Open challenges first, closed ones at the end. Stable otherwise.
        this.challenges = list.map((c, i) => ({ c, i }))
          .sort((a, b) => (!!a.c.closed - !!b.c.closed) || a.i - b.i).map((x) => x.c);
        this.rememberTitles(this.challenges);
      } catch (e) {
        this.listError = e.message;
      } finally {
        this.listLoading = false;
      }
    },

    applyFilters() {
      const q = new URLSearchParams();
      if (this.filters.track) q.set("track", this.filters.track);
      if (this.filters.tier) q.set("tier", this.filters.tier);
      const next = "#/" + (q.toString() ? "?" + q : "");
      // Replace, not push: changing a filter is not navigation.
      history.replaceState(null, "", next);
      this.loadChallenges();
    },

    setTier(t) { this.filters.tier = t; this.applyFilters(); },

    // ── challenge detail ──
    resetDetail() {
      this.detailReady = false;
      this.answer = "";
      this.result = null;
      this.submitError = "";
      this.fileNote = "";
      this.writeupMsg = "";
      this.writeupErr = "";
      this.writeupBody = "";
      this.hintErr = "";
      this.turnstileId = null;
    },

    async loadDetail(slug, quiet) {
      if (!quiet) { this.detailLoading = true; this.detailError = ""; }
      try {
        const d = await api("GET", "/api/challenges/" + encodeURIComponent(slug));
        if (this.route.name !== "detail" || this.route.slug !== slug) return;
        this.detail = d && typeof d === "object" ? d : {};
        this.detailReady = true;
        this.rememberTitles([d]);
        document.title = (d?.title || "Challenge") + " | Honne Arena";
        if (!quiet) {
          const mine = this.myWriteup;
          this.writeupBody = mine ? String(mine.body || "") : "";
          this.$nextTick(() => this.renderTurnstile());
        }
      } catch (e) {
        if (!quiet) this.detailError = e.message;
      } finally {
        if (!quiet) this.detailLoading = false;
      }
    },

    // Solve mode: solved by me, or by a teammate (one solve scores for the team).
    get solvedHere() { return !!(this.detail.solved_by_me || this.detail.solved_by_team); },
    get canWriteup() {
      if (!this.signedIn || this.needsHandle) return false;
      // Older servers may not send submitted_by_me; fall back to what we can see.
      if (typeof this.detail.submitted_by_me === "boolean") return this.detail.submitted_by_me;
      return !!(this.detail.solved_by_me || this.detail.my_best != null || this.myWriteup);
    },
    get cfg() { return (this.detail && typeof this.detail.config === "object" && this.detail.config) || {}; },
    get board() { return Array.isArray(this.detail?.board) ? this.detail.board : []; },
    get hints() { return Array.isArray(this.detail?.hints) ? this.detail.hints : []; },
    get writeups() { return Array.isArray(this.detail?.writeups) ? this.detail.writeups : []; },
    get samples() { return Array.isArray(this.cfg.samples) ? this.cfg.samples : []; },
    get boardHasAnswers() { return this.board.some((r) => r && r.answer != null && r.answer !== ""); },
    get myWriteup() {
      const h = this.me.user?.handle;
      return h ? this.writeups.find((w) => w.handle === h) : null;
    },
    get itemIdsHref() { return safeHref(this.cfg.item_ids); },

    // Kind-specific facts shown beside the description.
    get facts() {
      const c = this.cfg, k = this.detail?.kind, out = [];
      const add = (label, v) => { if (v !== undefined && v !== null && v !== "") out.push({ label, value: String(v) }); };
      if (k === "flag") add("Flag format", c.flag_format);
      if (k === "steering") {
        add("Model", c.model_id); add("Layer", c.layer); add("Token budget", c.token_budget);
        if (typeof c.higher_is_better === "boolean") add("Board", c.higher_is_better ? "Higher score wins" : "Lower score wins");
      }
      if (k === "predictions") add("Metric", c.metric === "accuracy" ? "Accuracy" : c.metric === "auroc" ? "AUROC" : c.metric);
      if (k === "code") {
        add("Language", c.language);
        add("Time limit", c.time_limit_s != null ? c.time_limit_s + " s" : null);
        add("Memory", c.memory_mb != null ? c.memory_mb + " MB" : null);
      }
      if (k === "circuits") {
        add("Model", c.model_id); add("Max heads", c.max_heads);
        if (c.n_layers != null && c.n_heads != null) add("Shape", c.n_layers + " layers x " + c.n_heads + " heads");
      }
      return out;
    },

    // ── answer helpers per kind ──
    get approxTokens() {
      // Rough guide only. The server counts with the model's own tokenizer.
      const t = this.answer.trim();
      if (!t) return 0;
      const words = t.split(/\s+/).length;
      return Math.max(words, Math.ceil(t.length / 4));
    },
    get tokenBudget() { return Number(this.cfg.token_budget) || null; },

    get heads() {
      const raw = this.answer.split(/[\s,;]+/).map((s) => s.trim()).filter(Boolean);
      const valid = [], invalid = [], seen = new Set();
      for (const h of raw) {
        if (!HEAD_RE.test(h)) { invalid.push(h); continue; }
        const [layer, head] = h.split(".").map((n) => parseInt(n, 10));
        const nl = Number(this.cfg.n_layers), nh = Number(this.cfg.n_heads);
        if ((nl && layer >= nl) || (nh && head >= nh)) { invalid.push(h); continue; }
        const norm = layer + "." + head;
        if (!seen.has(norm)) { seen.add(norm); valid.push(norm); }
      }
      return { valid, invalid };
    },
    get maxHeads() { return Number(this.cfg.max_heads) || null; },
    get headsOver() { return this.maxHeads != null && this.heads.valid.length > this.maxHeads; },

    get predLines() {
      return this.answer.split("\n").filter((l) => l.trim() && !l.trim().startsWith("#")).length;
    },

    get canSubmit() {
      if (!this.detailReady || this.submitting || this.detail.closed) return false;
      const k = this.detail.kind;
      if (k === "circuits") return this.heads.valid.length > 0 && !this.heads.invalid.length && !this.headsOver;
      return !!this.answer.trim();
    },

    // Tab in the code editor inserts spaces. Esc, then Tab, moves focus on as usual.
    codeKey(ev) {
      if (ev.key === "Escape") { this.escArmed = true; return; }
      if (ev.key !== "Tab" || ev.shiftKey || ev.ctrlKey || ev.altKey || ev.metaKey) {
        this.escArmed = false;
        return;
      }
      if (this.escArmed) { this.escArmed = false; return; }
      ev.preventDefault();
      const el = ev.target, s = el.selectionStart, e = el.selectionEnd, pad = "    ";
      el.setRangeText(pad, s, e, "end");
      this.answer = el.value;
    },

    readFile(ev) {
      const f = ev.target.files && ev.target.files[0];
      this.fileNote = "";
      if (!f) return;
      if (f.size > MAX_UPLOAD_BYTES) {
        this.fileNote = "That file is over 5 MB. Trim it or paste the lines instead.";
        ev.target.value = "";
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        this.answer = String(reader.result || "");
        this.fileNote = "Loaded " + f.name + " (" + this.predLines + " lines).";
      };
      reader.onerror = () => { this.fileNote = "Couldn't read that file."; };
      reader.readAsText(f);
    },

    async submit() {
      if (!this.canSubmit) return;
      const d = this.detail;
      let answer = this.answer;
      if (d.kind === "circuits") answer = this.heads.valid.join(", ");
      if (d.kind === "flag") answer = answer.trim();
      const body = { challenge: d.slug, answer };
      const tok = this.turnstileToken();
      if (tok) body.turnstile_token = tok;
      this.submitting = true;
      this.result = null;
      this.submitError = "";
      try {
        this.result = await api("POST", "/api/submit", body);
        if (d.kind === "flag" && this.result?.correct) this.answer = "";
        await Promise.all([this.loadDetail(d.slug, true), this.loadMe()]);
      } catch (e) {
        this.submitError = e.message;
        if (e.status === 401) await this.loadMe();
      } finally {
        this.submitting = false;
        this.resetTurnstile();
      }
    },

    // A plain description of the last result, for the aria-live region.
    get hasResult() { return !!this.result; },
    get resultView() {
      const r = this.result;
      if (!r) return { tone: "", title: "", big: "", lines: [] };
      const lines = [];
      let tone = "ok", title = "", big = "";
      if (r.verdict != null) {
        const v = String(r.verdict);
        title = VERDICTS[v.toLowerCase()] || v.replace(/_/g, " ");
        tone = /^(ac|accepted|ok)$/i.test(v) ? "ok" : "bad";
      }
      if (r.passed != null && r.total != null) lines.push("Passed " + r.passed + " of " + r.total + " tests.");
      if (r.mode === "solve" || typeof r.correct === "boolean") {
        if (!title) title = r.correct ? "Correct!" : "Not quite";
        tone = r.correct ? "ok" : "bad";
        if (r.first_blood) lines.unshift("First blood! Your team solved it first.");
        else if (r.correct) lines.unshift("Solved. Points go to your team.");
        else if (r.verdict == null) lines.unshift("That's not it. Check the format and try again.");
      }
      if (r.score != null) {
        if (!title) title = "Scored";
        big = this.fmtScore(r.score);
        tone = "ok";
      }
      const skip = new Set(["challenge", "mode", "correct", "first_blood", "score", "verdict", "passed", "total"]);
      const LABELS = { tokens: "Tokens", n: "Items scored", rank: "Rank", heads: "Heads",
                       faithfulness: "Faithfulness", ld_circuit: "Logit diff, your heads",
                       ld_full: "Logit diff, full model", ld_ablated: "Logit diff, all ablated" };
      for (const [k, v] of Object.entries(r)) {
        if (skip.has(k) || v == null || typeof v === "object") continue;
        const label = LABELS[k] || (k.charAt(0).toUpperCase() + k.slice(1)).replace(/_/g, " ");
        lines.push(label + ": " + String(v));
      }
      return { tone, title, big, lines };
    },

    // ── hints ──
    async unlockHint(h) {
      this.hintErr = "";
      if (!this.signedIn) { this.goSignIn(); return; }
      const learn = this.detail.tier === "learn" || !h.cost;
      const ok = await this.confirm(
        "Unlock hint " + h.position + "?",
        learn ? "Hints are free on learn challenges." :
          "This costs your team " + h.cost + " points. You can't undo it.",
        learn ? "Show hint" : "Unlock for " + h.cost + " points");
      if (!ok) return;
      try {
        const got = await api("POST", "/api/challenges/" + encodeURIComponent(this.detail.slug) +
          "/hints/" + encodeURIComponent(h.position) + "/unlock", {});
        const i = this.detail.hints.findIndex((x) => x.position === h.position);
        if (i >= 0 && got) this.detail.hints.splice(i, 1, Object.assign({}, h, got, { unlocked: true }));
      } catch (e) {
        this.hintErr = e.message;
      }
    },

    // ── writeups ──
    async postWriteup() {
      this.writeupMsg = "";
      this.writeupErr = "";
      const body = this.writeupBody.trim();
      if (!body) { this.writeupErr = "Write something first."; return; }
      if (body.length > 20000) { this.writeupErr = "Keep it under 20,000 characters."; return; }
      this.busy = true;
      try {
        await api("POST", "/api/challenges/" + encodeURIComponent(this.detail.slug) + "/writeup", { body });
        this.writeupMsg = "Posted. Thanks for sharing!";
        await this.loadDetail(this.detail.slug, true);
      } catch (e) {
        this.writeupErr = e.message;
      } finally {
        this.busy = false;
      }
    },

    // ── leaderboard ──
    get seasons() { return Array.isArray(this.season?.seasons) ? this.season.seasons : []; },

    selectSeason(id) {
      this.boardSeason = id == null || String(id) === String(this.season?.id) ? "" : String(id);
      return this.loadBoard();
    },
    isBoardSeason(id) {
      return this.boardSeason ? String(id) === String(this.boardSeason) : String(id) === String(this.season?.id);
    },

    async loadBoard() {
      this.boardError = "";
      const q = this.boardSeason ? "?season=" + encodeURIComponent(this.boardSeason) : "";
      try {
        this.seasonBoard = await api("GET", "/api/leaderboard" + q);
      } catch (e) {
        this.seasonBoard = null;
        this.boardError = e.message;
      }
    },

    get boardSeasonName() {
      const s = this.seasonBoard?.season;
      if (s && typeof s === "object") return s.name || "";
      if (typeof s === "string") return s;
      const id = this.boardSeason || this.season?.id;
      return this.seasons.find((x) => String(x.id) === String(id))?.name || this.season?.name || "";
    },
    get boardTeams() { return Array.isArray(this.seasonBoard?.teams) ? this.seasonBoard.teams : []; },
    get viewingActiveBoard() { return !this.boardSeason || String(this.boardSeason) === String(this.season?.id); },

    async loadAllTime() {
      this.allTimeError = "";
      try {
        const d = await api("GET", "/api/tracks");
        this.allTime = Array.isArray(d?.tracks) ? d.tracks : [];
        if (!this.allTimeTrack || !this.allTime.some((t) => t.track === this.allTimeTrack)) {
          this.allTimeTrack = this.allTime[0]?.track || "";
        }
      } catch (e) {
        this.allTimeError = e.message;
      }
    },
    get allTimeBoard() {
      const t = this.allTime.find((x) => x.track === this.allTimeTrack);
      return Array.isArray(t?.board) ? t.board : [];
    },

    // ── handle ──
    async saveHandle() {
      this.handleMsg = "";
      this.handleErr = "";
      const h = this.handleInput.trim();
      if (!HANDLE_RE.test(h)) {
        this.handleErr = "Use 1 to 32 characters: letters, numbers, spaces, and - _ .";
        return;
      }
      this.busy = true;
      try {
        const d = await api("POST", "/api/me/handle", { handle: h });
        if (d?.user) this.me.user = d.user;
        await this.loadMe();
        this.handleMsg = "Saved. You're playing as " + h + ".";
      } catch (e) {
        this.handleErr = e.message;
      } finally {
        this.busy = false;
      }
    },

    // ── teams ──
    async createTeam() {
      this.teamMsg = ""; this.teamErr = ""; this.createdCode = "";
      const name = this.teamName.trim();
      if (!name) { this.teamErr = "Give your team a name."; return; }
      this.busy = true;
      try {
        const d = await api("POST", "/api/teams", { name });
        this.createdCode = typeof d?.join_code === "string" ? d.join_code : "";
        this.teamMsg = "Team created. Share the join code with your teammates.";
        this.teamName = "";
        await this.loadMe();
      } catch (e) {
        this.teamErr = e.message;
      } finally {
        this.busy = false;
      }
    },

    async joinTeam() {
      this.teamMsg = ""; this.teamErr = ""; this.createdCode = "";
      const code = this.joinCode.trim();
      if (!code) { this.teamErr = "Paste the join code you were given."; return; }
      this.busy = true;
      try {
        const d = await api("POST", "/api/teams/join", { join_code: code });
        const name = typeof d?.team === "string" ? d.team : d?.team?.name || "the team";
        this.teamMsg = "You joined " + name + ".";
        this.joinCode = "";
        await this.loadMe();
      } catch (e) {
        this.teamErr = e.message;
      } finally {
        this.busy = false;
      }
    },

    get teamJoinCode() { return this.me.team?.join_code || this.createdCode || ""; },

    async copy(text) {
      try {
        await navigator.clipboard.writeText(text);
        this.copied = true;
        setTimeout(() => { this.copied = false; }, 2000);
      } catch (_) {
        this.copied = false;
      }
    },

    // ── sign in ──
    async requestLink() {
      this.signinErr = "";
      this.devLink = null;
      const e = this.email.trim();
      if (!/^[^\s@]+@northeastern\.edu$/i.test(e)) {
        this.signinErr = "Use your @northeastern.edu email.";
        return;
      }
      this.busy = true;
      try {
        const d = await api("POST", "/api/auth/request", { email: e });
        this.signinSent = true;
        this.devLink = safeHref(d?.dev_link);
      } catch (err) {
        this.signinErr = err.message;
      } finally {
        this.busy = false;
      }
    },

    async signOut() {
      try { await api("POST", "/api/auth/logout", {}); } catch (_) { /* clear locally anyway */ }
      this.me = { user: null, team: null, tiers: {} };
      this.signinSent = false;
      this.email = "";
      this.toast("You're signed out.");
      location.hash = "#/";
    },

    // ── admin ──
    async loadAdmin() {
      this.admin.error = "";
      try {
        const [c, u] = await Promise.all([
          api("GET", "/api/admin/challenges"),
          api("GET", "/api/admin/usage").catch(() => null),
        ]);
        this.admin.challenges = Array.isArray(c?.challenges) ? c.challenges : [];
        this.rememberTitles(this.admin.challenges);
        this.admin.usage = u;
      } catch (e) {
        this.admin.error = e.message;
      }
    },

    async createSeason() {
      const s = this.admin.season;
      s.msg = ""; s.err = "";
      if (!s.name.trim()) { s.err = "Name the season."; return; }
      if (s.active && !(await this.confirm("Make this the active season?",
        "New submissions will count toward it. The current season's board stays readable.", "Create season"))) return;
      this.busy = true;
      try {
        await api("POST", "/api/admin/seasons", { name: s.name.trim(), active: !!s.active });
        s.msg = "Season created.";
        s.name = "";
        await this.loadSeason();
        await this.loadAdmin();
      } catch (e) {
        s.err = e.message;
      } finally {
        this.busy = false;
      }
    },

    setKind(kind) {
      const f = this.admin.form;
      f.kind = kind;
      f.configText = JSON.stringify(CONFIG_TEMPLATES[kind] || {}, null, 2);
    },

    get configParse() {
      const t = this.admin.form.configText.trim();
      if (!t) return { ok: true, value: {} };
      try {
        const v = JSON.parse(t);
        if (!v || typeof v !== "object" || Array.isArray(v)) return { ok: false, error: "Config must be a JSON object, like {\"key\": 1}." };
        return { ok: true, value: v };
      } catch (e) {
        return { ok: false, error: "Not valid JSON: " + e.message };
      }
    },

    // JSON config with the typed kind fields laid over it.
    get mergedConfig() {
      const f = this.admin.form, k = f.k, p = this.configParse;
      if (!p.ok) return null;
      const c = Object.assign({}, p.value);
      const num = (v) => (v === "" || v == null ? undefined : Number(v));
      const set = (key, v) => { if (v !== undefined && v !== "" && !(typeof v === "number" && Number.isNaN(v))) c[key] = v; };
      if (f.kind === "flag") { set("flag_format", k.flag_format.trim() || undefined); c.case_sensitive = !!k.case_sensitive; }
      if (f.kind === "predictions") set("metric", k.metric);
      if (f.kind === "steering") {
        set("model_id", k.model_id.trim() || undefined); set("layer", num(k.layer));
        set("token_budget", num(k.token_budget)); c.higher_is_better = !!k.higher_is_better;
      }
      if (f.kind === "code") {
        set("language", k.language.trim() || undefined); set("time_limit_s", num(k.time_limit_s)); set("memory_mb", num(k.memory_mb));
      }
      if (f.kind === "circuits") set("max_heads", num(k.max_heads));
      return c;
    },

    addHintRow() { this.admin.form.hints.push({ body: "", cost: this.admin.form.tier === "learn" ? 0 : 50 }); },
    removeHintRow(i) { this.admin.form.hints.splice(i, 1); },

    async createChallenge() {
      const f = this.admin.form;
      this.admin.formMsg = ""; this.admin.formErr = "";
      if (!/^[a-z0-9][a-z0-9-]{0,63}$/.test(f.slug)) { this.admin.formErr = "Slug: lowercase letters, digits and dashes, starting with a letter or digit."; return; }
      if (!f.title.trim()) { this.admin.formErr = "Add a title."; return; }
      if (!this.configParse.ok) { this.admin.formErr = this.configParse.error; return; }
      if (f.kind === "flag" && !f.flag.trim()) { this.admin.formErr = "A flag challenge needs its flag."; return; }
      const body = {
        slug: f.slug, title: f.title.trim(), kind: f.kind, track: f.track, tier: f.tier,
        description: f.description, config: this.mergedConfig,
        points_initial: Number(f.points_initial), points_min: Number(f.points_min),
        points_decay: Number(f.points_decay), hidden: !!f.hidden,
      };
      if (f.kind === "flag") body.flag = f.flag;
      if (f.closes_at) {
        const dt = new Date(f.closes_at);
        if (Number.isNaN(dt.getTime())) { this.admin.formErr = "Closing time isn't a valid date."; return; }
        body.closes_at = dt.toISOString();
      }
      const hints = f.hints.filter((h) => h.body.trim()).map((h) => ({ body: h.body.trim(), cost: Number(h.cost) || 0 }));
      if (hints.length) body.hints = hints;
      this.busy = true;
      try {
        const d = await api("POST", "/api/admin/challenges", body);
        this.admin.formMsg = "Created " + (d?.slug || f.slug) + (f.hidden ? " (hidden until you unhide it)." : ".");
        this.admin.form = blankChallengeForm();
        await this.loadAdmin();
      } catch (e) {
        this.admin.formErr = e.message;
      } finally {
        this.busy = false;
      }
    },

    async patchChallenge(c, patch, ask) {
      this.admin.error = "";
      if (ask && !(await this.confirm(ask.title, ask.body, ask.ok))) return;
      try {
        await api("PATCH", "/api/admin/challenges/" + encodeURIComponent(c.slug), patch);
        await this.loadAdmin();
        this.toast("Updated " + c.slug + ".");
      } catch (e) {
        this.admin.error = e.message;
      }
    },
    toggleHidden(c) { return this.patchChallenge(c, { hidden: !c.hidden }); },
    closeNow(c) {
      return this.patchChallenge(c, { closes_at: new Date().toISOString() },
        { title: "Close " + c.title + " now?", body: "No more submissions. Writeups open for people who submitted.", ok: "Close it" });
    },
    reopen(c) { return this.patchChallenge(c, { closes_at: null }); },

    async addHintTo() {
      const a = this.admin;
      a.hintMsg = "";
      if (!a.hintFor || !a.hintBody.trim()) { a.hintMsg = "Pick a challenge and write the hint."; return; }
      try {
        const d = await api("POST", "/api/admin/challenges/" + encodeURIComponent(a.hintFor) + "/hints",
          { body: a.hintBody.trim(), cost: Number(a.hintCost) || 0 });
        a.hintMsg = "Added as hint " + (d?.position ?? "") + ".";
        a.hintBody = "";
      } catch (e) {
        a.hintMsg = e.message;
      }
    },

    // ── live updates ──
    connectLive(now) {
      if (this._ws && (this._ws.readyState === 0 || this._ws.readyState === 1)) return;
      clearTimeout(this._wsTimer);
      if (now) this._wsTries = 0;
      let ws;
      try {
        ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
      } catch (_) {
        this.scheduleLive();
        return;
      }
      this._ws = ws;
      this.live = "connecting";
      ws.onopen = () => { this.live = "on"; this._wsTries = 0; };
      ws.onmessage = (ev) => this.onLive(ev.data);
      ws.onerror = () => { try { ws.close(); } catch (_) { /* already closed */ } };
      ws.onclose = () => {
        if (this._ws === ws) this._ws = null;
        this.live = "off";
        this.scheduleLive();
      };
    },

    // Exponential backoff with jitter: about 1 s, 2 s, 4 s ... capped at 60 s.
    scheduleLive() {
      clearTimeout(this._wsTimer);
      const base = Math.min(60000, 1000 * 2 ** Math.min(this._wsTries, 6));
      this._wsTries++;
      const wait = base / 2 + Math.random() * base / 2;
      this._wsTimer = setTimeout(() => {
        if (document.visibilityState === "hidden") return; // reconnect on return instead
        this.connectLive();
      }, wait);
    },

    onLive(raw) {
      let m;
      try { m = JSON.parse(raw); } catch (_) { return; }
      if (!m || typeof m !== "object") return;
      const slug = typeof m.challenge === "string" ? m.challenge : "";
      const handle = typeof m.handle === "string" ? m.handle : "someone";
      const mine = this.me.user?.handle && handle === this.me.user.handle;
      if (m.type === "solve") {
        if (!mine) this.toastSolve(slug, handle, !!m.first_blood);
      } else if (m.type !== "entry") {
        return;
      }
      this._dirty.add(slug);
      clearTimeout(this._dirtyTimer);
      this._dirtyTimer = setTimeout(() => this.refreshDirty(), 700);
    },

    // Refetch only what the visible page shows, and only if the event touches it.
    refreshDirty() {
      const slugs = this._dirty;
      this._dirty = new Set();
      const r = this.route.name;
      if (r === "list") {
        if (this.challenges.some((c) => slugs.has(c.slug))) this.loadChallenges();
      } else if (r === "detail" && slugs.has(this.route.slug)) {
        this.loadDetail(this.route.slug, true);
      } else if (r === "leaderboard") {
        if (this.viewingActiveBoard) this.loadBoard();
        this.loadAllTime();
      }
    },

    async toastSolve(slug, handle, firstBlood) {
      if (!this._titles[slug] && !this._titlesTried) {
        this._titlesTried = true; // one list fetch at most, then fall back to the slug
        try {
          const d = await api("GET", "/api/challenges");
          this.rememberTitles(d?.challenges);
        } catch (_) { /* fall back to the slug */ }
      }
      const title = this.challengeTitle(slug);
      this.toast(firstBlood ? "First blood! " + handle + " solved " + title + "."
        : handle + " solved " + title + ".", firstBlood ? "gold" : "");
    },

    rememberTitles(list) {
      if (!Array.isArray(list)) return;
      for (const c of list) if (c && typeof c.slug === "string" && typeof c.title === "string") this._titles[c.slug] = c.title;
    },

    challengeTitle(slug) { return this._titles[slug] || slug || "a challenge"; },

    toast(text, tone) {
      const id = ++this._toastId;
      this.toasts.push({ id, text: String(text), tone: tone || "" });
      if (this.toasts.length > 4) this.toasts.shift();
      setTimeout(() => this.dismissToast(id), 7000);
    },
    dismissToast(id) { this.toasts = this.toasts.filter((t) => t.id !== id); },

    // ── confirm dialog (native <dialog>: focus trap and Esc for free) ──
    confirm(title, body, ok) {
      return new Promise((resolve) => {
        this.dialog = { title, body, ok: ok || "OK", resolve };
        const d = this.$refs.dialog;
        if (d && typeof d.showModal === "function") {
          d.showModal();
        } else {
          resolve(window.confirm(title + "\n\n" + body));
        }
      });
    },
    closeDialog(answer) {
      const d = this.$refs.dialog;
      const r = this.dialog.resolve;
      this.dialog.resolve = null;
      if (d?.open) d.close();
      if (r) r(!!answer);
    },

    // ── Turnstile (only loaded when the season turns CAPTCHA on) ──
    loadTurnstile() {
      if (document.getElementById("cf-turnstile-js")) return;
      const s = document.createElement("script");
      s.id = "cf-turnstile-js";
      s.src = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
      s.async = true;
      s.onload = () => this.renderTurnstile();
      document.head.appendChild(s);
    },
    renderTurnstile() {
      const el = document.getElementById("turnstile-slot");
      if (!el || !window.turnstile || !this.season?.captcha_sitekey || el.childElementCount) return;
      try { this.turnstileId = window.turnstile.render(el, { sitekey: this.season.captcha_sitekey, theme: "dark" }); } catch (_) { /* widget unavailable */ }
    },
    turnstileToken() {
      if (!window.turnstile || this.turnstileId == null) return "";
      try { return window.turnstile.getResponse(this.turnstileId) || ""; } catch (_) { return ""; }
    },
    resetTurnstile() {
      if (window.turnstile && this.turnstileId != null) { try { window.turnstile.reset(this.turnstileId); } catch (_) { /* ignore */ } }
    },

    // ── misc ──
    toggleScanlines() {
      this.scanlines = !this.scanlines;
      storeSet("local", SCAN_KEY, this.scanlines ? "on" : "off");
    },

    md(s) { return window.HonneMD.render(s, { headingBase: 3 }); },
    trackName(id) { return TRACKS.find((t) => t.id === id)?.name || String(id || ""); },
    kindName(k) { return KIND_NAMES[k] || String(k || ""); },
    tierLabel(t) { return t === "learn" ? "Learn" : t === "compete" ? "Compete" : String(t || ""); },
    tierOf(track) {
      const t = this.me.tiers && this.me.tiers[track];
      return t && typeof t === "object" ? t : { tier: 0, rank: null, points: 0 };
    },
    tierBadge(n) { return n >= 1 && n <= 3 ? "T" + n : ""; },

    fmtScore(v) {
      if (v === null || v === undefined || v === "") return "none";
      const n = Number(v);
      if (!Number.isFinite(n)) return String(v);
      // Metrics such as AUROC live in [0, 1], so keep 4 decimals there even for 1 or 0.
      if (Math.abs(n) >= 10) return Number.isInteger(n) ? n.toLocaleString() : n.toFixed(Math.abs(n) >= 100 ? 1 : 2);
      return n.toFixed(4);
    },
    fmtNum(v) { const n = Number(v); return Number.isFinite(n) ? n.toLocaleString() : "0"; },
    fmtDate(s) {
      if (!s) return "";
      const d = new Date(s);
      if (Number.isNaN(d.getTime())) return String(s);
      return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
    },
    fmtRel(s) {
      const d = new Date(s);
      if (!s || Number.isNaN(d.getTime())) return "";
      const diff = (d.getTime() - Date.now()) / 1000;
      const abs = Math.abs(diff);
      const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
      if (abs < 3600) return rtf.format(Math.round(diff / 60), "minute");
      if (abs < 86400) return rtf.format(Math.round(diff / 3600), "hour");
      return rtf.format(Math.round(diff / 86400), "day");
    },
    isMe(handle) { return !!this.me.user?.handle && handle === this.me.user.handle; },
    isMyTeam(team) { return !!this.me.team?.name && team === this.me.team.name; },
  };
}

window.arena = arena;
