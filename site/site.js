/* liberdex.net: the replay in the hero, the configurator, the theme switch.
   Plain JavaScript, no build step, nothing fetched from anywhere. */
(function () {
  "use strict";
  const $ = (s, r) => (r || document).querySelector(s);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const wait = (ms) => new Promise((r) => setTimeout(r, reduced ? 0 : ms));

  /* ------------------------------------------------------------- theme */
  const themeBtn = $("#theme");
  function paintTheme() {
    const light = document.documentElement.dataset.theme === "light";
    themeBtn.textContent = light ? "dark" : "light";
    themeBtn.setAttribute("aria-label", light ? "Switch to dark theme" : "Switch to light theme");
  }
  themeBtn.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("liberdex.theme", next); } catch (e) { /* fine */ }
    paintTheme();
  });
  paintTheme();

  const menu = $("#menu");
  $("#menu-toggle").addEventListener("click", () => {
    const open = menu.classList.toggle("open");
    $("#menu-toggle").setAttribute("aria-expanded", String(open));
  });
  menu.addEventListener("click", (e) => { if (e.target.tagName === "A") menu.classList.remove("open"); });

  // The imprint's contact lines ship as base64 of the reversed text, so a
  // harvester reading the HTML finds no address to match. To change them, pipe
  // the new text through
  //   python3 -c 'import base64,sys;print(base64.b64encode(sys.stdin.read().strip()[::-1].encode()).decode())'
  // A blank line starts a paragraph; a word with an @ becomes a mail link.
  document.querySelectorAll("[data-contact]").forEach((box) => {
    const bytes = Uint8Array.from(atob(box.dataset.contact), (c) => c.charCodeAt(0));
    const text = [...new TextDecoder().decode(bytes)].reverse().join("");
    box.replaceChildren(...text.split("\n\n").map((para) => {
      const p = el("p");
      para.split("\n").forEach((line, i) => {
        if (i) p.append(el("br"));
        line.split(/(\S+@\S+)/).forEach((part, j) => {
          if (j % 2 === 0) { p.append(part); return; }
          const a = el("a", null, part);
          a.href = "mailto:" + part;
          p.append(a);
        });
      });
      return p;
    }));
  });

  // The legal pages share the masthead and nothing below it.
  if (!$("#demo")) return;

  /* ------------------------------------------------------------- replay */
  const demos = window.LIBERDEX_DEMO || [];
  const input = $("#q");
  const serp = $("#serp");
  const tally = $("#tally");
  const chips = $("#examples");
  let run = 0;

  demos.forEach((d) => {
    const b = el("button", null, d.query);
    b.type = "button";
    b.dataset.id = d.id;
    b.addEventListener("click", () => play(d));
    const li = el("li");
    li.append(b);
    chips.append(li);
  });

  function pressed(id) {
    chips.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.id === id)));
  }

  function hit(h, i) {
    const row = el("article", "hit");
    row.style.setProperty("--rel", h.rel.toFixed(3));
    const meta = el("div", "meta");
    meta.append(el("span", "dot"), el("span", "site", h.site));
    const h3 = el("h3");
    const a = el("a", null, h.title);
    a.href = h.url;
    a.rel = "noreferrer nofollow";
    a.addEventListener("click", (e) => e.preventDefault());
    a.title = "Illustrative: " + h.site + " is a reserved name, not a site";
    h3.append(a);
    const bar = el("div", "bar");
    bar.title = "relevance " + h.rel.toFixed(2);
    bar.append(el("i"));
    row.append(meta, h3, el("p", null, h.text), bar);
    return row;
  }

  async function play(d) {
    const mine = ++run;
    pressed(d.id);
    input.value = d.query;
    serp.replaceChildren();
    tally.textContent = "";
    for (let i = 0; i < d.hits.length; i++) {
      const g = el("div", "ghost");
      g.setAttribute("aria-hidden", "true");
      g.style.cssText = "margin:0 0 22px;opacity:.45";
      const s1 = el("span"); s1.style.cssText = "display:block;width:38%;height:11px;margin-bottom:9px;border-radius:3px;background:var(--surface)";
      const s2 = el("span"); s2.style.cssText = "display:block;width:72%;height:17px;margin-bottom:9px;border-radius:3px;background:var(--surface)";
      const s3 = el("span"); s3.style.cssText = "display:block;width:100%;height:11px;border-radius:3px;background:var(--surface)";
      g.append(s1, s2, s3);
      serp.append(g);
    }
    const t = d.tick;
    tally.textContent = "asking the model where the answer lives";
    await wait(700);
    if (mine !== run) return;
    tally.textContent = "first page requested at " + t.firstUrl + " ms, model still naming pages";
    await wait(800);
    if (mine !== run) return;
    tally.textContent = "reading " + t.dispatched + " pages in parallel";
    await wait(900);
    if (mine !== run) return;
    serp.replaceChildren();
    tally.textContent = t.read + " pages read in full, ranked in " + t.rank + " ms, " + (t.total / 1000).toFixed(1) + " s from question to answer";
    if (d.answer) {
      const box = el("div", "answer");
      const p = el("p", null, d.answer.text + " ");
      d.answer.cites.forEach((n) => p.append(el("span", "cite", String(n))));
      box.append(p);
      serp.append(box);
      requestAnimationFrame(() => box.classList.add("in"));
      await wait(200);
    }
    for (let i = 0; i < d.hits.length; i++) {
      if (mine !== run) return;
      const row = hit(d.hits[i], i);
      serp.append(row);
      requestAnimationFrame(() => row.classList.add("in"));
      await wait(160);
    }
  }

  $("#search").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value.trim().toLowerCase();
    const d = demos.find((x) => x.query.toLowerCase() === q) || demos[0];
    play(d);
  });
  $("#replay").addEventListener("click", () => {
    const cur = chips.querySelector("button[aria-pressed=true]");
    play(demos.find((x) => x.id === (cur && cur.dataset.id)) || demos[0]);
  });

  // Start when the demo scrolls into view; on a short screen that is at once.
  if (demos.length) {
    const io = new IntersectionObserver((es) => {
      if (es.some((x) => x.isIntersecting)) { io.disconnect(); play(demos[0]); }
    }, { threshold: 0.2 });
    io.observe($("#demo"));
  }

  // The benchmark bars grow once, the first time the figure is on screen.
  const bench = $("#bench");
  if (bench && !reduced) {
    bench.classList.add("wait");
    const bo = new IntersectionObserver((es) => {
      if (es.some((x) => x.isIntersecting)) {
        bo.disconnect();
        requestAnimationFrame(() => bench.classList.remove("wait"));
      }
    }, { threshold: 0.3 });
    bo.observe(bench);
  }

  /* ------------------------------------------------------- configurator */
  const PROVIDERS = {
    openrouter: { key: "OPENROUTER_API_KEY", model: "google/gemini-3.5-flash-lite", label: "OpenRouter" },
    openai:     { key: "OPENAI_API_KEY", model: "gpt-5-mini", label: "OpenAI" },
    gemini:     { key: "GEMINI_API_KEY", model: "", label: "Google Gemini" },
    groq:       { key: "GROQ_API_KEY", model: "", label: "Groq" },
    together:   { key: "TOGETHER_API_KEY", model: "", label: "Together" },
    mistral:    { key: "MISTRAL_API_KEY", model: "", label: "Mistral" },
    deepseek:   { key: "DEEPSEEK_API_KEY", model: "", label: "DeepSeek" },
    xai:        { key: "XAI_API_KEY", model: "", label: "xAI" },
    anthropic:  { key: "ANTHROPIC_API_KEY", model: "", label: "Anthropic (OpenAI-compatible shim)" },
    ollama:     { key: "", model: "qwen3:8b", label: "Ollama (local)", local: true },
    lmstudio:   { key: "", model: "", label: "LM Studio (local)", local: true },
    llamacpp:   { key: "", model: "", label: "llama.cpp server (local)", local: true },
    vllm:       { key: "", model: "", label: "vLLM (local)", local: true },
    "claude-cli":   { key: "", model: "opus", label: "Claude Code subscription (claude)", cli: true },
    "codex-cli":    { key: "", model: "", label: "Codex subscription (codex)", cli: true },
    "gemini-cli":   { key: "", model: "", label: "Gemini CLI subscription (gemini)", cli: true },
    "opencode-cli": { key: "", model: "", label: "OpenCode subscription (opencode)", cli: true },
  };
  // No PyPI release yet: uvx takes the package from the repository.
  const UVX = "uvx --from git+https://github.com/beneglo/liberdex liberdex";
  const HOSTS = {
    "claude-code": "Claude Code", codex: "Codex", opencode: "OpenCode", cursor: "Cursor",
    gemini: "Gemini CLI", hermes: "hermes-agent", openclaw: "OpenClaw", t3code: "T3 Code",
  };

  const form = $("#config");
  const provSel = $("#provider");
  const hostSel = $("#host");
  Object.entries(PROVIDERS).forEach(([k, v]) => provSel.append(new Option(v.label, k)));
  Object.entries(HOSTS).forEach(([k, v]) => hostSel.append(new Option(v, k)));

  function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;"); }
  function state() {
    const f = new FormData(form);
    return {
      where: f.get("where"), host: f.get("host"), provider: f.get("provider"),
      model: (f.get("model") || "").trim(),
      keys: (f.get("keys") || "").trim(), cors: (f.get("cors") || "").trim(),
      run: f.get("run"),
    };
  }

  function render() {
    const s = state();
    const p = PROVIDERS[s.provider];
    const agent = s.where === "agent";
    const endpoint = s.where === "endpoint";
    form.querySelectorAll("[data-when]").forEach((n) => {
      n.hidden = !n.dataset.when.split(" ").includes(s.where);
    });
    $("#model").placeholder = p.model || (p.cli ? "the CLI's own default" : "model id");
    $("#model-row").hidden = agent;
    $("#provider-row").hidden = agent;

    const env = [];
    const c = (t) => '<span class="c">' + esc(t) + "</span>";
    const k = (t) => '<span class="k">' + esc(t) + "</span>";
    let runCmd = "";
    let then = "";
    if (agent) {
      runCmd = UVX + " install " + s.host;
      then = "Restart " + HOSTS[s.host] + " and search. " +
        (s.host === "hermes" ? "In hermes, web search now is liberdex." :
         "The host model does the thinking, so there is no key to set.");
      env.push(c("# Optional, only if liberdex should also search without the host:"));
      env.push(c("# OPENROUTER_API_KEY=..."));
    } else {
      const spec = s.provider + "/" + (s.model || p.model);
      if (p.key) env.push(k(p.key) + "=" + c("...paste your key..."));
      if (p.cli && !s.model && !p.model) env.push(c("# " + s.provider + " uses whatever model its own config names."));
      env.push(k("LIBERDEX_PLANNER_MODEL") + "=" + (s.model || p.model ? spec : s.provider + "/"));
      if (p.cli) env.push(c("# A subscription CLI answers whole, 10 to 20 s in: use the `deep` setting."));
      if (p.local && !p.key) env.push(c("# " + PROVIDERS[s.provider].label + " is reached at its usual localhost port; set LIBERDEX_BASE_URL to change it."));
      if (endpoint) {
        env.push(k("LIBERDEX_KEYS") + "=" + (s.keys ? esc(s.keys) : c("...comma-separated keys callers must present...")));
        if (s.cors) env.push(k("LIBERDEX_CORS") + "=" + esc(s.cors));
        else env.push(c("# LIBERDEX_CORS=https://app.example   origins a browser may call from"));
        env.push(c("# LIBERDEX_MAX_INFLIGHT=8   searches at once; past it, 503 + Retry-After"));
      }
      if (endpoint && s.run === "docker") {
        runCmd = "docker run -p 8080:8080 --env-file .env \\\n  -v liberdex-cache:/home/liberdex/.cache/liberdex \\\n  ghcr.io/beneglo/liberdex";
        then = "The image ships with the ranking models, so <b>/health</b> is green within seconds. The JSON API is at <b>/search</b>, MCP at <b>/mcp</b>.";
      } else if (endpoint) {
        runCmd = UVX + " serve --host 0.0.0.0";
        then = "Put the .env next to where you run it, or export the lines. <b>liberdex warmup</b> downloads the two ranking models (130 MB) ahead of the first search.";
      } else {
        runCmd = UVX + " serve --open";
        then = "Opens <b>http://127.0.0.1:8080</b>. The same choices live in the page's settings drawer, which writes them to <b>~/.config/liberdex/</b>; the .env is for people who prefer a file.";
      }
    }
    $("#out-run").textContent = runCmd;
    $("#out-env").innerHTML = env.join("\n");
    $("#out-then").innerHTML = then;
  }
  form.addEventListener("input", render);
  form.addEventListener("change", render);
  render();

  document.querySelectorAll(".copy").forEach((b) => {
    b.addEventListener("click", async () => {
      const src = $(b.dataset.copy);
      try {
        await navigator.clipboard.writeText(src.textContent);
        b.classList.add("did"); b.textContent = "copied";
        setTimeout(() => { b.classList.remove("did"); b.textContent = "copy"; }, 1400);
      } catch (e) {
        b.textContent = "select and copy";
      }
    });
  });
})();
