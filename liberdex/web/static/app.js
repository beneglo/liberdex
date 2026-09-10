/* liberdex: the search page.

   One form, two views. `data-view` on <body> moves the same input between the
   lens composition on the home page and the masthead above a SERP; nothing is
   duplicated, so there is only ever one query and one place it lives.

   Everything the server sends is inserted as text. Titles and passages come off
   arbitrary web pages, which makes them the least trustworthy strings in the
   product; the only markup this file builds, it builds as nodes. */
"use strict";

const $ = (sel) => document.querySelector(sel);
const body = document.body;
const form = $("#search");
const input = $("#q");
const out = $("#out");
const depth = $("#depth");

const STORE = "liberdex.";
const DEPTHS = ["fast", "standard", "deep"];

let inflight = null;   // AbortController for the search on the wire
let last = null;       // the reply on screen, so a re-render costs no fetch
let waking = false;    // a /health poll is already running

/* Result pages. There is no result set to slice: liberdex has no index, so a
   page past the first is a new round. The planner is asked, over every page
   already shown, for what comes next, and none of the shown pages is fetched
   or returned again. Each page's reply is kept, so going back costs nothing
   and going forward sends the server the whole "already shown" list. */
const MAX_PAGE = 10;
const pages = { key: "", replies: [] };   // replies[i] is page i+1

function pageOf(search) {
  const p = parseInt(new URLSearchParams(search).get("p") || "1", 10);
  return p >= 1 && p <= MAX_PAGE ? p : 1;
}
function pageKey(q) { return currentDepth() + "\u0000" + q; }
function pagesFor(q) {
  if (pages.key !== pageKey(q)) { pages.key = pageKey(q); pages.replies = []; }
  return pages.replies;
}
function shownBefore(replies, page) {
  const urls = [];
  for (let i = 0; i < page - 1 && i < replies.length; i++) {
    for (const r of (replies[i] && replies[i].results) || []) urls.push(r.url);
  }
  return urls;
}
function href(q, page) {
  return "/?q=" + encodeURIComponent(q) + (page > 1 ? "&p=" + page : "");
}

/* ------------------------------------------------------------------ storage */
function remember(key, value) {
  try { localStorage.setItem(STORE + key, value); } catch (e) { /* no store */ }
}
function recall(key, fallback) {
  try { return localStorage.getItem(STORE + key) ?? fallback; }
  catch (e) { return fallback; }
}

/* -------------------------------------------------------------------- state */
function currentDepth() {
  const on = depth.querySelector("input:checked");
  return on ? on.value : "standard";
}
function wantsAnswer() {
  return $("#answer-toggle").getAttribute("aria-pressed") === "true";
}
function queryOf(search) {
  return (new URLSearchParams(search).get("q") || "").trim();
}

/* ------------------------------------------------------------------ drawing */
function clear() { out.replaceChildren(); }

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

function notice(text, strong) {
  const n = el("div", "notice");
  if (strong) n.append(el("b", null, strong), " ");
  n.append(document.createTextNode(text));
  return n;
}

function skeleton() {
  clear();
  out.setAttribute("aria-busy", "true");
  for (let i = 0; i < 8; i++) {
    const g = el("div", "ghost");
    for (const c of ["g1", "g2", "g3", "g4"]) g.append(el("span", c));
    out.append(g);
  }
}

function iconSrc(r) {
  if (r.favicon && r.favicon.startsWith("data:")) return r.favicon;
  const p = new URLSearchParams({ site: r.site || "" });
  if (r.favicon) p.set("u", r.favicon);
  return "/icon?" + p.toString();
}

/* The synthesize prompt cites with the page's number in square brackets, and
   those numbers are 1-based against `results`, so each one can become a link to
   the row it names. The prompt asks for one bracket per page ([1][4]) and
   models write [1, 4] about as often, so both are read. A number outside the
   SERP is left as the text it is rather than linked somewhere wrong. */
const CITE = /\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]/g;

function withCitations(text, count) {
  const frag = document.createDocumentFragment();
  let at = 0;
  for (const m of text.matchAll(CITE)) {
    const nums = m[1].split(",").map((n) => parseInt(n, 10));
    if (!nums.every((n) => n >= 1 && n <= count)) continue;
    // "antigen [1, 4]." reads as "antigen¹⁴.": the space before the marker
    // belongs to the bracket notation, not to the sentence.
    if (m.index > at) {
      frag.append(document.createTextNode(
        text.slice(at, m.index).replace(/ $/, "")));
    }
    for (const n of nums) {
      const a = el("a", "cite", String(n));
      a.href = "#hit-" + n;
      a.title = "Jump to result " + n;
      frag.append(a);
    }
    at = m.index + m[0].length;
  }
  frag.append(document.createTextNode(text.slice(at)));
  return frag;
}

/* The answer arrives as prose that is lightly Markdown-shaped: it names
   identifiers in backticks, and it reaches for a bullet list when the answer
   has parts. Rendered literally that reads as a bug, so the two markers it
   actually uses are honoured, and only those. Not a Markdown renderer, and it
   builds nodes rather than markup, because the text it is formatting was
   written by a model reading arbitrary web pages. */
const INLINE = /`([^`\n]+)`|\*\*([^*\n]+)\*\*/g;
const BULLET = /^\s*[-*\u2022]\s+/;

function inline(text, count) {
  const frag = document.createDocumentFragment();
  let at = 0;
  for (const m of text.matchAll(INLINE)) {
    if (m.index > at) frag.append(withCitations(text.slice(at, m.index), count));
    frag.append(el(m[1] !== undefined ? "code" : "strong", null, m[1] ?? m[2]));
    at = m.index + m[0].length;
  }
  frag.append(withCitations(text.slice(at), count));
  return frag;
}

function answerBody(text, count) {
  const frag = document.createDocumentFragment();
  let para = [];
  let list = null;

  const flush = () => {
    if (!para.length) return;
    const p = el("p");
    p.append(inline(para.join(" "), count));
    frag.append(p);
    para = [];
  };

  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) { flush(); list = null; continue; }
    if (BULLET.test(line)) {
      flush();
      if (!list) { list = el("ul"); frag.append(list); }
      const li = el("li");
      li.append(inline(line.replace(BULLET, ""), count));
      list.append(li);
      continue;
    }
    // A bullet run written on one line ("...requirements: * a * b") is the
    // same list with the newlines lost, and splitting it back out beats
    // printing the asterisks.
    if (list === null && / \* /.test(line)) {
      const [head, ...items] = line.split(/ \* /);
      if (head.trim()) para.push(head.trim());
      flush();
      list = el("ul");
      frag.append(list);
      for (const item of items) {
        const li = el("li");
        li.append(inline(item.trim(), count));
        list.append(li);
      }
      continue;
    }
    if (list) { list = null; }
    para.push(line);
  }
  flush();
  return frag;
}

function answerCard(reply) {
  const card = el("div", "answer");
  const body = el("div", "body");
  body.append(answerBody(reply.answer, reply.results.length));
  card.append(body);
  const model = reply.stats && reply.stats.answer_model;
  if (model) card.append(el("p", "by", "answered by " + model));
  return card;
}

/* A synthesized answer can run to 900 tokens, which on a 640px column pushes
   all ten links off the screen, so the answer eats the SERP it was drawn from.
   Clipped only when it actually overflows, so a two-line answer keeps no
   control it does not need. */
function clipAnswer(card) {
  const body = card.querySelector(".body");
  if (!body) return;
  // Clamp first, then ask whether it overflowed: with no max-height in force
  // scrollHeight and clientHeight are the same number and nothing ever clips.
  card.classList.add("clipped");
  if (body.scrollHeight <= body.clientHeight + 4) {
    card.classList.remove("clipped");
    return;
  }
  const more = el("button", "more", "Show the rest of the answer");
  more.type = "button";
  more.setAttribute("aria-expanded", "false");
  more.addEventListener("click", () => {
    const open = card.classList.toggle("open");
    more.setAttribute("aria-expanded", String(open));
    more.textContent = open ? "Collapse the answer" : "Show the rest of the answer";
    if (!open) card.scrollIntoView({ block: "start", behavior: "auto" });
  });
  card.insertBefore(more, card.querySelector(".by"));
}

function hit(r, i) {
  const row = el("article", "hit");
  row.id = "hit-" + (i + 1);
  if (r.debug && r.debug.unverified) row.classList.add("unverified");

  const meta = el("div", "meta");
  const img = el("img");
  img.src = iconSrc(r);
  img.alt = "";
  img.loading = "lazy";
  img.width = 14; img.height = 14;
  meta.append(img, el("span", "site", r.site || new URL(r.url).hostname));
  if (r.published) meta.append(el("span", "sep", "·"), el("span", null, r.published));
  // Where the row came from: the planner's memory, a site's own search, or a
  // hub's links. No index engine can say.
  if (r.source) meta.append(el("span", "sep", "·"), el("span", null, r.source));

  const h = el("h2");
  // A page liberdex could not read has no title either, and the extractor falls
  // back to the host, which makes two different papers on one site look like
  // the same row. The URL is the honest label there.
  const bare = !r.title || r.title === r.site;
  const a = el("a", null, bare ? r.url : r.title);
  a.href = r.url;
  a.rel = "noreferrer";
  h.append(a);

  const dead = unread(r);
  const body_ = el("p", dead ? "dead" : null,
    dead ? ("Couldn't read this page: "
            + (r.status && r.status !== 200 ? "HTTP " + r.status
                                            : "it returned no readable text"))
         : snippet(r));

  // The relevance the ranker settled on, drawn as one of the mark's index lines
  // lighting up as far as the page earned. `score` is the other number and it
  // is the wrong one to draw: it orders this SERP but means nothing across two.
  const rel = Math.max(0, Math.min(1, r.relevance || 0));
  const bar = el("div", "bar");
  const fill = el("i");
  fill.style.transform = "scaleX(" + rel.toFixed(3) + ")";
  bar.append(fill);
  bar.title = "relevance " + rel.toFixed(2);

  row.append(meta, h, body_, bar);
  return row;
}

/* Reference markers a page carries in its own prose, Wikipedia's [1] and its
   [edit] links, survive extraction, and in a three-line preview they are most
   of what you see. Stripped only where several appear, so the one page that
   genuinely writes `xs[0]` keeps it. */
const REFS = /\[(?:\d{1,3}|edit)\]/g;

function readable(text) {
  const hits = text.match(REFS);
  if (hits && hits.length > 1) text = text.replace(REFS, "");
  return text.replace(/\s+/g, " ").trim();
}

/* The passages come back in page order, each scored by how much of the query it
   covers. A reader gets the best one; the whole set still went to the answer.
   A window taken from the middle of a page says so, the way a quotation does. */
function snippet(r) {
  const parts = r.passages || [];
  if (!parts.length) return "";
  const scores = r.passage_scores || [];
  let best = 0;
  for (let i = 1; i < parts.length; i++) {
    if ((scores[i] || 0) > (scores[best] || 0)) best = i;
  }
  const text = readable(parts[best]);
  return best > 0 && text ? "\u2026" + text : text;
}

/* Some pages come back as a bot wall: a 203 with a JavaScript challenge and no
   prose, which the extractor reduces to the hostname. Printing that as a
   snippet would be inventing a summary of a page nobody read, so the row says
   what actually happened instead. */
function unread(r) {
  const bare = snippet(r).replace(/^\u2026/, "").replace(/\.$/, "").trim();
  if (!bare) return true;
  if (bare === (r.site || "").replace(/\.$/, "")) return true;
  return bare.length < 40 && !bare.includes(" ");
}

function pager(reply, replies) {
  const page = reply.page || 1;
  const nav = el("nav", "pager");
  nav.setAttribute("aria-label", "Result pages");
  const prev = el("button", "step", "\u2190 Page " + (page - 1));
  prev.type = "button";
  prev.disabled = page <= 1;
  prev.addEventListener("click", () => turn(page - 1));
  const list = el("ol");
  const known = Math.max(replies.length, page);
  for (let i = 1; i <= known; i++) {
    const li = el("li");
    const b = el("button", null, String(i));
    b.type = "button";
    if (i === page) b.setAttribute("aria-current", "page");
    b.addEventListener("click", () => { if (i !== page) turn(i); });
    li.append(b);
    list.append(li);
  }
  const next = el("button", "step", "Page " + (page + 1) + " \u2192");
  next.type = "button";
  next.disabled = page >= MAX_PAGE || !reply.results.length;
  next.addEventListener("click", () => turn(page + 1));
  nav.append(prev, list, next);
  const hint = el("p", "hint",
    "Each page is a new round: the planner is asked for what it has not shown "
    + "yet, and the pages above are never fetched twice.");
  const wrap = el("div", "pages");
  wrap.append(nav, hint);
  return wrap;
}

function render(reply) {
  last = reply;
  clear();
  out.setAttribute("aria-busy", "false");
  const stats = reply.stats || {};
  const page = reply.page || 1;

  if (stats.degraded) {
    // Nothing was read, so the rows are the planner's recollection and any
    // answer was composed over that recollection. Saying only "these rows"
    // would leave the most confident thing on the page uncovered.
    out.append(notice(
      "No page could be fetched. These rows"
      + (reply.answer && wantsAnswer() ? ", and the answer below them," : "")
      + " are what the planner recalled, not what liberdex read.",
      "Unverified."));
  } else if (stats.planner_error === "cancelled") {
    // Cancelled is not failed: the plan's line protocol front-loads routes and
    // hubs, so a planner cut off mid-stream has usually contributed something.
    // Saying it "didn't answer" would misreport a partial plan as no plan, and
    // send the reader looking for an API key that is not the problem.
    const got = stats.candidates_llm || 0;
    const n = el("div", "notice");
    n.append(el("b", null, "The planner ran out of time."), " ",
             document.createTextNode(
               got ? `Only ${got} of its URLs arrived before the deadline; the `
                     + "rest of this page came from the sites' own search and "
                     + "indexes. "
                   : "None of its URLs arrived before the deadline, so this "
                     + "page came from the sites' own search and indexes alone. "));
    // The fix is one control away, so it is offered rather than described.
    if (currentDepth() !== "deep") {
      const act = el("button", "act", "Search again with deep");
      act.type = "button";
      act.addEventListener("click", () => {
        const deep = depth.querySelector('input[value="deep"]');
        deep.checked = true;
        remember("depth", "deep");
        run(queryOf(location.search) || input.value);
      });
      n.append(act);
    }
    out.append(n);
  } else if (stats.planner_error) {
    out.append(notice(
      "These came from site-native search alone. Check the planner model and "
      + "its API key. " + stats.planner_error, "The planner didn't answer."));
  }
  if (stats.answer_error && wantsAnswer()) {
    out.append(notice(String(stats.answer_error), "No answer:"));
  }

  if (!reply.results.length) {
    out.append(notice(
      page > 1 ? "The planner had nothing left to add on this page."
               : "Try deep, or name a site with site:example.com.",
      page > 1 ? "No more pages." : "Nothing came back for that."));
    if (page > 1) out.append(pager(reply, pagesFor(reply.query)));
    return;
  }

  let card = null;
  if (reply.answer && wantsAnswer()) {
    card = answerCard(reply);
    out.append(card);
  }

  const ms = (reply.timings && reply.timings.total_ms) || 0;
  const bits = [];
  if (page > 1) bits.push("page " + page);
  bits.push(reply.results.length + (reply.results.length === 1 ? " result" : " results"));
  if (stats.fetched_ok) bits.push(stats.fetched_ok + " pages read");
  // A second planning pass ran: deep, or a page turn. Worth a word, because
  // it is the one thing that distinguishes this SERP from a standard one.
  if (stats.rounds === 2) bits.push("2 rounds");
  if (ms) bits.push((ms / 1000).toFixed(1) + "s");
  out.append(el("p", "tally", bits.join(" · ")));

  reply.results.forEach((r, i) => out.append(hit(r, i)));
  out.append(pager(reply, pagesFor(reply.query)));
  // After the card is in the document, so scrollHeight means something.
  if (card) clipAnswer(card);
}

/* ----------------------------------------------------------------- searching */
function parseSites(q) {
  // `site:example.com` is the one operator people type without being told, so
  // it is honoured, as a dispatch allowlist: what include_domains does, and
  // the only thing an index-free engine can do with it.
  const sites = [];
  const rest = q.replace(/\bsite:(\S+)/gi, (_, d) => { sites.push(d); return ""; });
  return { query: rest.replace(/\s+/g, " ").trim() || q, sites };
}

async function run(q, page) {
  page = page || 1;
  const replies = pagesFor(q);
  // A page can only follow the pages before it: the server needs every URL
  // already shown. Landing cold on ?p=3 starts at page one instead.
  if (page > 1 && replies.length < page - 1) {
    history.replaceState({ q, page: 1 }, "", href(q, 1));
    page = 1;
  }
  if (inflight) inflight.abort();
  const ctl = new AbortController();
  inflight = ctl;

  document.title = q + (page > 1 ? " — page " + page : "") + " — liberdex";
  body.dataset.view = "results";
  input.value = q;
  skeleton();

  const { query, sites } = parseSites(q);
  const payload = {
    query, top_k: 10, depth: currentDepth(),
    include_favicon: true, include_debug: true,
  };
  if (sites.length) payload.include_domains = sites;
  if (page > 1) {
    payload.page = page;
    payload.exclude_urls = shownBefore(replies, page);
  } else if (wantsAnswer()) {
    // The question is answered once, on page one.
    payload.answer = "synthesize";
  }

  let resp;
  try {
    resp = await api("/search", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
      signal: ctl.signal,
    });
  } catch (e) {
    if (ctl.signal.aborted) return;
    clear();
    out.setAttribute("aria-busy", "false");
    out.append(notice("Is `liberdex serve` still running?", "Couldn't reach liberdex."));
    return;
  }
  if (ctl.signal.aborted) return;

  if (resp.status === 503) { await wake(q, page); return; }
  if (resp.status === 401) {
    // This server asks callers for a key. Once; it is kept for next time.
    if (askKey()) { run(q, page); return; }
    clear();
    out.setAttribute("aria-busy", "false");
    out.append(notice(apiKey() ? "This liberdex refused the API key it was given."
                               : "This liberdex asks for an API key and none was given.",
                      "Not signed in."));
    return;
  }
  if (!resp.ok) {
    let why = "HTTP " + resp.status;
    try {
      const err = await resp.json();
      if (typeof err.detail === "string") why = err.detail;
      else if (Array.isArray(err.detail)) why = err.detail.map((d) => d.msg).join("; ");
    } catch (e) { /* not JSON; the status line is the whole story */ }
    clear();
    out.setAttribute("aria-busy", "false");
    out.append(notice(why, "liberdex refused that search:"));
    return;
  }
  const reply = await resp.json();
  reply.page = page;
  replies[page - 1] = reply;
  // Turning a page forgets what lay beyond it: those were planned over a
  // different "already shown" list than this reply now makes.
  replies.length = page;
  render(reply);
  inflight = null;
  window.scrollTo({ top: 0, behavior: "auto" });
}

/* Show a page already fetched, or fetch it. Pushes history either way, so the
   browser's back button walks the pages. */
function turn(page) {
  const q = queryOf(location.search) || input.value.trim();
  if (!q || page < 1 || page > MAX_PAGE) return;
  history.pushState({ q, page }, "", href(q, page));
  show(q, page);
}

function show(q, page) {
  const replies = pagesFor(q);
  const kept = replies[page - 1];
  if (kept) {
    if (inflight) { inflight.abort(); inflight = null; }
    document.title = q + (page > 1 ? " — page " + page : "") + " — liberdex";
    body.dataset.view = "results";
    input.value = q;
    render(kept);
    window.scrollTo({ top: 0, behavior: "auto" });
    return;
  }
  run(q, page);
}

/* The engine loads ~130 MB of ranking models on first start, and answers 503
   until it has. That is a wait, not a failure, so the page says so and comes
   back on its own. */
async function wake(q, page) {
  if (waking) return;
  waking = true;
  clear();
  out.setAttribute("aria-busy", "true");
  out.append(notice("It downloads and loads its ranking models once. "
                    + "This search will run by itself when they are ready.",
                    "liberdex is still starting."));
  const until = Date.now() + 180000;
  while (Date.now() < until) {
    await new Promise((r) => setTimeout(r, 2000));
    try {
      const h = await fetch("/health");
      if (h.ok && (await h.json()).ok) { waking = false; run(q, page); return; }
    } catch (e) { /* still down */ }
  }
  waking = false;
  clear();
  out.setAttribute("aria-busy", "false");
  out.append(notice("Check the terminal running `liberdex serve`.",
                    "liberdex didn't finish starting."));
}

function go(q, push) {
  q = q.trim();
  if (!q) return;
  if (push) history.pushState({ q, page: 1 }, "", href(q, 1));
  run(q, 1);
}

function home() {
  if (inflight) { inflight.abort(); inflight = null; }
  body.dataset.view = "home";
  document.title = "liberdex";
  clear();
  last = null;
  input.value = "";
  input.focus();
}

/* -------------------------------------------------------------------- wiring */
form.addEventListener("submit", (e) => {
  e.preventDefault();
  input.blur();
  go(input.value, true);
});

depth.addEventListener("change", () => {
  remember("depth", currentDepth());
  const q = queryOf(location.search);
  // Another depth is another search; its pages start over from one.
  if (q) { history.replaceState({ q, page: 1 }, "", href(q, 1)); run(q, 1); }
});

$("#answer-toggle").addEventListener("click", (e) => {
  const on = e.currentTarget.getAttribute("aria-pressed") !== "true";
  e.currentTarget.setAttribute("aria-pressed", String(on));
  remember("answer", on ? "1" : "0");
  const q = queryOf(location.search);
  if (!q) return;
  // Turning it off only hides a card we already paid for. Turning it on needs
  // an answer, and asks for one only when this reply has none, and only on
  // page one, where the answer lives.
  if (!on || (last && last.answer) || (last && last.page > 1)) { if (last) render(last); }
  else run(q, 1);
});

$("#theme-toggle").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  applyTheme(next);
  remember("theme", next);
});

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const b = $("#theme-toggle");
  b.setAttribute("aria-label",
    theme === "light" ? "Switch to dark theme" : "Switch to light theme");
}

document.querySelector(".lockup").addEventListener("click", (e) => {
  e.preventDefault();
  history.pushState({ q: "" }, "", "/");
  home();
});

// A citation scrolls to its row and says which one it landed on, because a jump
// with no acknowledgement reads as a page that moved on its own.
out.addEventListener("click", (e) => {
  const cite = e.target.closest("a.cite");
  if (!cite) return;
  const row = document.getElementById(cite.hash.slice(1));
  if (!row) return;
  e.preventDefault();
  row.scrollIntoView({ block: "center",
    behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  row.classList.add("flash");
  setTimeout(() => row.classList.remove("flash"), 900);
});

window.addEventListener("popstate", () => {
  const q = queryOf(location.search);
  if (q) { input.value = q; show(q, pageOf(location.search)); } else home();
});

/* ----------------------------------------------------------------- the key
   A server bound beyond loopback may ask callers for a key (LIBERDEX_KEYS).
   The page keeps it in localStorage, like the theme: it is this reader's own
   key for this server, and nothing else reads it. */
function apiKey() { return recall("key", ""); }

function askKey() {
  const k = window.prompt("This liberdex asks for an API key.");
  if (!k) return false;
  remember("key", k.trim());
  return true;
}

async function api(path, opts) {
  opts = opts || {};
  const headers = Object.assign({}, opts.headers || {});
  const k = apiKey();
  if (k) headers.authorization = "Bearer " + k;
  return fetch(path, Object.assign({}, opts, { headers }));
}

/* ------------------------------------------------------------------ settings
   The drawer. Served only when the server is on loopback (or told to): it
   writes model choices and keys to the machine running liberdex. */
const drawer = $("#settings");
let settings = null;      // the last GET /settings

function profileOf(spec) {
  // "openrouter/google/x" -> ["openrouter", "google/x"]; "ollama" -> ["ollama", ""]
  if (!spec) return ["", ""];
  const i = spec.indexOf("/");
  if (i < 0) return [spec, ""];
  return [spec.slice(0, i), spec.slice(i + 1)];
}

function profileRow(name) {
  return (settings && settings.profiles || []).find((p) => p.name === name) || null;
}

function badge(p) {
  if (p.kind === "claude_cli" || p.kind === "cli") return p.on_path ? "on this machine" : "not installed";
  if (p.local) return "local";
  return p.has_key ? "key set" : "needs a key";
}

function fillProfiles(select, keepEmpty) {
  const current = select.value;
  select.replaceChildren();
  if (keepEmpty) {
    const none = el("option", null, "same as planner");
    none.value = "";
    select.append(none);
  }
  for (const p of settings.profiles) {
    const o = el("option", null, `${p.name} — ${badge(p)}`);
    o.value = p.name;
    select.append(o);
  }
  select.value = current;
}

function fillKeyRow(role) {
  const row = $(`#s-${role}-key`);
  const name = $(`#s-${role}-profile`).value;
  const p = profileRow(name);
  const wantsKey = p && p.kind === "openai" && !p.local && p.api_key_env.length;
  row.hidden = !wantsKey;
  if (!wantsKey) return;
  row.querySelector(".env").textContent = p.api_key_env[0];
  row.querySelector("input").value = "";
  row.querySelector("input").placeholder = p.has_key ? "replace key" : "paste key";
  row.querySelector(".have").hidden = !p.has_key;
}

async function fillModels(role) {
  const name = $(`#s-${role}-profile`).value;
  const list = $(`#s-${role}-models`);
  list.replaceChildren();
  const p = profileRow(name);
  if (!p || p.kind !== "openai" || !(p.has_key || p.local)) return;
  try {
    const r = await api("/settings/models?profile=" + encodeURIComponent(name));
    if (!r.ok) return;
    const ids = (await r.json()).models || [];
    for (const id of ids.slice(0, 400)) {
      const o = el("option", null, id);
      o.value = id;
      list.append(o);
    }
  } catch (e) { /* the picker is a convenience */ }
}

function fillStatus() {
  const st = $("#s-status");
  st.replaceChildren();
  const pl = settings.roles.planner;
  if (pl.resolved) {
    st.append("planner ", el("b", null, pl.resolved), " (" + pl.why + ")");
  } else {
    st.append(el("span", "bad", pl.why || "no planner"));
  }
  st.append(document.createElement("br"));
  st.append("answer ", el("b", null, settings.roles.answer.resolved || "—"));
  if (settings.planner_error) {
    st.append(document.createElement("br"));
    st.append(el("span", "bad", settings.planner_error));
  }
}

function fillSettings() {
  fillStatus();
  const [pp, pm] = profileOf(settings.roles.planner.configured);
  const [ap, am] = profileOf(settings.roles.answer.configured);
  fillProfiles($("#s-planner-profile"), false);
  fillProfiles($("#s-answer-profile"), true);
  // An unconfigured planner shows what auto picked, so the drawer never opens blank.
  const [rp, rm] = profileOf(settings.roles.planner.resolved);
  $("#s-planner-profile").value = pp || rp || "openrouter";
  $("#s-planner-model").value = pp ? pm : (pp === "" && !settings.roles.planner.configured ? "" : pm);
  $("#s-answer-profile").value = ap;
  $("#s-answer-model").value = am;
  fillKeyRow("planner"); fillKeyRow("answer");
  fillModels("planner"); fillModels("answer");
  fillCommons();
  $("#s-note").textContent = "";
}

// Shown once a shared plan cache has issued this machine a key, not before.
function fillCommons() {
  const c = settings.commons || {};
  $("#s-commons").hidden = !c.registered;
  $("#s-commons-note").textContent = "A lookup sends the question's content words, sorted, "
    + "not the question as typed; a plan this machine wrote is offered back under them. "
    + "Registered at " + c.url + ".";
}

async function loadSettings() {
  const r = await api("/settings");
  if (!r.ok) return false;
  settings = await r.json();
  fillSettings();
  return true;
}

function specOf(role) {
  const p = $(`#s-${role}-profile`).value;
  const m = $(`#s-${role}-model`).value.trim();
  if (!p) return "";
  return m ? p + "/" + m : p;
}

async function saveSettings(e) {
  e.preventDefault();
  const body = { planner_model: specOf("planner"), answer_model: specOf("answer"), keys: {} };
  for (const role of ["planner", "answer"]) {
    const row = $(`#s-${role}-key`);
    if (row.hidden) continue;
    const v = row.querySelector("input").value.trim();
    if (v) body.keys[row.querySelector(".env").textContent] = v;
  }
  const save = $("#s-save");
  save.disabled = true;
  $("#s-note").textContent = "saving …";
  try {
    const r = await api("/settings", { method: "PUT",
      headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) {
      let why = "HTTP " + r.status;
      try { why = (await r.json()).detail || why; } catch (x) { /* status is the story */ }
      $("#s-note").textContent = why;
      return;
    }
    settings = await r.json();
    fillSettings();
    $("#s-note").textContent = "saved to " + settings.config.models;
    hint();
  } finally {
    save.disabled = false;
  }
}

async function openSettings() {
  if (!(await loadSettings())) return;
  drawer.showModal();
}

/* The home view says, in one line, when nothing on this machine can plan.
   A search would still run on site-native routes and look like a thin result
   rather than a missing model. */
async function hint() {
  const h = $("#hint");
  h.hidden = true;
  h.replaceChildren();
  let r;
  try { r = await api("/settings"); } catch (e) { return; }
  if (!r.ok) return;                      // not served on this bind: no drawer, no hint
  $("#settings-toggle").hidden = false;
  settings = await r.json();
  if (settings.roles.planner.resolved && !settings.planner_error) return;
  const b = el("button", null, "choose a model");
  b.type = "button";
  b.addEventListener("click", openSettings);
  h.append("No planner is set up, so a search would run on site-native routes alone. ", b, ".");
  h.hidden = false;
}

$("#settings-toggle").addEventListener("click", openSettings);
$("#settings-close").addEventListener("click", () => drawer.close());
$("#settings-form").addEventListener("submit", saveSettings);
for (const role of ["planner", "answer"]) {
  $(`#s-${role}-profile`).addEventListener("change", () => {
    $(`#s-${role}-model`).value = "";
    fillKeyRow(role); fillModels(role);
  });
}

/* --------------------------------------------------------------------- start */
(function start() {
  applyTheme(document.documentElement.dataset.theme === "light" ? "light" : "dark");

  const d = recall("depth", "standard");
  if (DEPTHS.includes(d)) depth.querySelector(`input[value="${d}"]`).checked = true;

  const wantAnswer = recall("answer", "1") === "1";
  $("#answer-toggle").setAttribute("aria-pressed", String(wantAnswer));

  const tmpl = $("#tmpl");
  if (tmpl) tmpl.textContent = location.origin + "/?q=%s";

  const q = queryOf(location.search);
  if (q) { input.value = q; show(q, pageOf(location.search)); } else home();
  hint();
})();
