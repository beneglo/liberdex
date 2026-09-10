# liberdex inside an agent CLI

Claude Code, Codex, OpenCode, Cursor, Gemini CLI, hermes-agent, OpenClaw, T3
Code: one command each.

```bash
uvx --from git+https://github.com/beneglo/liberdex liberdex install claude-code
uvx --from git+https://github.com/beneglo/liberdex liberdex install codex
uvx --from git+https://github.com/beneglo/liberdex liberdex install opencode
uvx --from git+https://github.com/beneglo/liberdex liberdex install cursor
uvx --from git+https://github.com/beneglo/liberdex liberdex install gemini
uvx --from git+https://github.com/beneglo/liberdex liberdex install hermes
uvx --from git+https://github.com/beneglo/liberdex liberdex install openclaw
uvx --from git+https://github.com/beneglo/liberdex liberdex install t3code
```

Each writes two things and prints every path it touches: the `liberdex` MCP
server into the host's own config, and the skill into the host's skills
directory. Run it again and it changes nothing, and `--dry-run` shows the plan.
Restart the host; the tools are `search` and `extract`. hermes is the one host
that gets liberdex as its own `web_search` instead; see below.

Claude Code can also take it as a plugin, which keeps the skill and the server
together and updates them together:

```
/plugin marketplace add beneglo/liberdex
/plugin install liberdex
```

## No key, no second model

The host is already a strong model with the web in its memory, and you are
already paying for it. So the tool asks the host to write the plan, meaning the
URLs it recalls, the per-site queries and the hub pages, and passes it as
`plan`. liberdex fetches, reads and ranks. There is no second LLM call, no API
key, and the first fetch leaves the moment the tool is called:

```
plan_first_url_ms: 0.1   stats.planner: "supplied"   # or "cache" / "commons": no model ran
```

The skill (`skills/liberdex/SKILL.md`, copied into the host) is what makes the
host's plans good: which sources to name first, never to guess an opaque
identifier, how to write a per-site query, when to reach for liberdex instead
of the host's built-in search. The tool description carries the short form so
the host does the right thing even before the skill is activated.

A host that would rather not plan omits `plan`, and liberdex plans with its own
configured model. On a machine with no model configured and no key, the tool
says exactly that and how to fix it (`liberdex install --model
openrouter/google/gemini-3.5-flash-lite`), rather than returning the thin
routes-only page that would look like a search.

## The tools

**`search`**

| argument | default | |
| --- | --- | --- |
| `query` | | the question, in the user's words |
| `plan` | none | `{candidates, routes, hubs, expansions, intent, lang}`; see [plan-protocol.md](../skills/liberdex/references/plan-protocol.md) |
| `top_k` | 8 | |
| `depth` | `standard` | `fast` 2.5 s, `standard` 6 s, `deep` 24 s with a second planning round |
| `include_domains`, `exclude_domains` | | gate what is fetched, not a post-filter |
| `freshness` | | `day`, `week`, `month`, `year` |
| `intent` | | pins the ranking profile; the plan's own is used otherwise |
| `token_budget` | 6000 | cap on the whole result, so a search does not flood the context |
| `include_text` | false | whole pages, not passages |
| `page`, `exclude_urls` | 1 | page two is a new round over what was shown |

The text result is numbered pages with `relevance` (0..1, comparable across
queries), `source` (`llm` for the host's own candidates, then `route:<site>`,
`hub`, `expand`, `deep`, `sitesearch`, `sitemap`, `refine`), the size of the
whole page in chars, and the passages
about the query. A page shipped with `include_text` and cut to the budget ends
in a note saying how much was shown and the `extract` call that reads the rest,
so the model never has to guess whether ` ...` was the page's own or the cut.
`structuredContent` is the same `SearchReply` the HTTP API returns.

**`extract`** takes `urls` (up to 20), an optional `query` (passages about it
instead of the whole page), `format` as `markdown` or `text`,
`token_budget_per_page` (default 4000), and `include_links`. Each page reports
its size in chars; one cut to the budget ends in the same note.

## Running it by hand

```bash
liberdex mcp                              # stdio, what the hosts start
liberdex mcp --planner-model ollama/qwen3:8b   # for searches that arrive without a plan
```

The same server is mounted at `/mcp` on `liberdex serve`, so a hosted liberdex
is a connector for claude.ai, Claude Desktop and ChatGPT; see
[deploy.md](deploy.md).

## Where the host has no web search of its own

OpenCode ships `websearch` only on its own Zen provider or with
`OPENCODE_ENABLE_EXA=1` / `OPENCODE_ENABLE_PARALLEL=1`; on your own keys it
has `webfetch` and nothing to find URLs with. T3 Code has none. Claude Code,
Codex and Gemini CLI have one, tied to their vendor. In all of them liberdex is
a second tool beside whatever is there, and the skill says to reach for it
first: full-page passages, the URLs the model already recalls as input, and it
runs on your machine.

## hermes-agent

hermes has a web search of its own with pluggable backends, so liberdex
becomes one: `web.backend: liberdex` in `~/.hermes/config.yaml`, and the
`web_search` and `web_extract` tools the agent already knows are liberdex.
There is no plan argument in that interface, so the provider writes the plan
with hermes's own model, borrowed through the plugin API (`ctx.llm`): the
model you already run, no key, one extra completion of a few hundred tokens
before the fetches start.

```bash
uvx --from git+https://github.com/beneglo/liberdex liberdex install hermes
```

installs liberdex into hermes's own Python, enables the plugin
(`hermes plugins enable liberdex`), sets the backend, and copies the skill to
`~/.hermes/skills/liberdex/`. `hermes tools` shows the backend;
`LIBERDEX_HERMES_DEPTH=deep` makes every search a research one.

hermes's own installer pins Python 3.11 and liberdex needs 3.12. On that
install, and with `--mcp`, the command wires the MCP server instead:
`hermes mcp add liberdex --command <liberdex> --args mcp`, a process of its own
on any Python, tools `mcp_liberdex_search` and `mcp_liberdex_extract` beside
hermes's `web_search`, and the plan written inside the tool call as in every
other host. `/reload-mcp` in a running session. The skill can also be taken
straight from the repository:

```
hermes skills install https://raw.githubusercontent.com/beneglo/liberdex/main/skills/liberdex/SKILL.md
```

## OpenClaw

OpenClaw's `web_search` needs a keyed provider and never falls back to a
free one, so on a fresh install there is no web search until a key is pasted.
`uvx --from git+https://github.com/beneglo/liberdex liberdex install openclaw` adds the MCP server to `~/.openclaw/openclaw.json`
(`openclaw config set` when the CLI is there, a plain-JSON edit otherwise; a
config with comments is left for you, with the snippet printed) and copies the
skill to `~/.openclaw/skills/liberdex/` and `~/.agents/skills/liberdex/`, which
OpenClaw reads too. `openclaw mcp doctor liberdex --probe` checks the server.

```json5
mcp: { servers: { liberdex: { command: "/path/to/liberdex", args: ["mcp"], enabled: true } } }
```

## T3 Code

T3 Code owns no tool config: it runs Claude Code, Codex and OpenCode and reads
their MCP configs. `uvx --from git+https://github.com/beneglo/liberdex liberdex install t3code` runs those three installers.
Two things it needs that the installer does: the command is an absolute path
(a relative one fails under T3 Code's working directory), and nothing depends
on the shell environment, because T3 Code started from the Dock has none.
With the plan supplied by the host model, liberdex needs no key at all.

## Skills directories, per host

| host | MCP config | skill |
| --- | --- | --- |
| Claude Code | `claude mcp add --scope user liberdex -- uvx --from git+https://github.com/beneglo/liberdex liberdex mcp` | `~/.claude/skills/liberdex/` |
| Codex | `codex mcp add liberdex -- uvx --from git+https://github.com/beneglo/liberdex liberdex mcp` (or `~/.codex/config.toml`) | `~/.codex/skills/liberdex/` and `~/.agents/skills/liberdex/` |
| OpenCode | `~/.config/opencode/opencode.json` `mcp.liberdex` | `~/.config/opencode/skills/liberdex/` |
| Cursor | `~/.cursor/mcp.json` | `~/.cursor/skills/liberdex/` |
| Gemini CLI | `~/.gemini/settings.json` | `~/.gemini/skills/liberdex/` |
| hermes-agent | `web.backend: liberdex` + `plugins.enabled` in `~/.hermes/config.yaml`; or `mcp_servers.liberdex` | `~/.hermes/skills/liberdex/` |
| OpenClaw | `~/.openclaw/openclaw.json` `mcp.servers.liberdex` | `~/.openclaw/skills/liberdex/` and `~/.agents/skills/liberdex/` |
| T3 Code | the Claude Code, Codex and OpenCode entries above | theirs |

The skill follows the [Agent Skills](https://agentskills.io) format, so any
host that reads `SKILL.md` can use it; `--command` overrides how the host
starts the server (default: this install's `liberdex`, else uvx from the repository).
