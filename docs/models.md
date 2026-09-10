# Models: planner and answer

## Planner backends

| backend | class | notes |
| --- | --- | --- |
| supplied | `planner.SuppliedPlanner` | the fastest one: the caller wrote the plan (the `plan` field on `/search`, the MCP tool's `plan`, `liberdex search --plan-file`). No LLM call; the first fetch leaves at t=0. What an agent CLI does. |
| OpenAI-compatible | `planner.OpenAIPlanner` | the intended hosted one. OpenRouter, vLLM, Groq, Together, and every local runtime. Streams. Defaults to `google/gemini-3.5-flash-lite`, thinking off. |
| Claude CLI | `planner.ClaudeCLIPlanner` | shells out to `claude -p`. No API key, streams, ~3.5s before the first token. |
| subscription CLI | `planner.CLIPlanner` | `codex-cli`, `gemini-cli`, `opencode-cli`, or any command in a `[providers.*]` block with `kind = "cli"`. No API key, no streaming: the plan lands whole, 10-20s in. Honest at `deep`, cut off at `fast`. |
| none | `planner=None` | site-native routes only. No LLM, ~0.6s, thin SERPs. |
| replay | `cache.ReplayPlanner` | re-emits a recorded plan with its recorded timing, so a search reruns without a planner. |

Which one runs: `--planner auto`, the default everywhere, takes in order
the spec you configured (`--planner-model`, `LIBERDEX_PLANNER_MODEL`,
`[models.planner]` in the config file), else the first hosted provider with a
key in the environment, else the `claude` binary if it is on PATH. With none of
those, the CLI stops and says so, the server starts and says so on the page,
and the MCP server serves searches that arrive with a `plan` and refuses the
ones that do not. None of them returns a routes-only SERP that looks like a
working search.

## Running the whole engine on the Claude CLI

`LIBERDEX_PLANNER=claude` needs no API key, and pointing the answer role at the
same binary means the engine runs with none at all:

```bash
liberdex serve --planner-model claude-cli/opus --answer-model claude-cli/sonnet
# the same, as environment: LIBERDEX_PLANNER_MODEL=claude-cli/opus ...
```

This is also what `auto` picks on a machine with the `claude` binary and no
API key, so on such a machine `liberdex serve` alone does it.

Both roles pass `--restricted`, which makes the CLI ignore the operator's own
`CLAUDE.md`, settings, hooks and output styles. Without it the answer is written
in whatever voice that machine's editor is configured for, and a personal
instruction file can steer an answer this API presents as grounded in the pages
liberdex fetched.

The CLI needs about three seconds to reach its first URL. That wait is not
taken out of the fetch window: each tier grants the planner a grace for its
first URL on top of its window (`fast` 1s, `standard` 4.5s, `deep` 6s;
`PLAN_GRACE` in `api.py`) and hands back whatever the planner did not use the
moment the first URL lands. A cached plan costs nothing. `fast` still cuts this
backend, three seconds being past a 1s grace: the planner is cancelled
mid-stream, `stats` reports `planner_error: cancelled`, and the search page
reports a cancelled planner as a partial plan and offers a one-click rerun at
`deep`.

The plan cache means the second run of a query answers from
`~/.cache/liberdex/plans.sqlite3` with no CLI call at all, so a repeated search
is complete even at `standard`. Only cold queries pay.

Environment for the server: `LIBERDEX_PLANNER` (`auto` | `claude` | `openai` |
`replay` | `none`), `LIBERDEX_PLANNER_MODEL` and `LIBERDEX_ANSWER_MODEL`
(`profile/model`), `LIBERDEX_BASE_URL`, the provider key names
(`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, ...), `LIBERDEX_REASONING_EFFORT`,
`LIBERDEX_PLAN_CACHE=0` to disable plan caching. `liberdex serve` takes the
same as flags, and the page's settings drawer writes them to
`~/.config/liberdex/models.toml` and `~/.config/liberdex/env`.

## Picking the planner model

The planner is a recall task on a fixed ~2.2k-token prompt that emits a ~400
token plan. Output throughput decides when the last URL gets dispatched and
time to first token decides when the first one does; intelligence scores move
neither. A fast, cheap model with thinking off is the right pick, which is why
`google/gemini-3.5-flash-lite` is the default.

## Thinking budget

Reasoning tokens are billed against `max_tokens`, so an unbounded thinking
budget truncates the plan itself. `LIBERDEX_REASONING_EFFORT` (or
`--reasoning-effort`) takes `none`, `minimal`, `low`, `medium`, `high`, and
defaults to `minimal` on any endpoint that understands the field at all.
`none` sends `reasoning: {"enabled": false}`, which some endpoints refuse
outright ("Reasoning is mandatory for this endpoint and cannot be disabled"), so
it is available rather than assumed. On those endpoints an explicit low level is
the fallback. Leaving the field unset entirely is the trap: a model that thinks
by default spends the budget before its first URL and returns a shorter plan.

The answer role has its own knob, `LIBERDEX_ANSWER_REASONING_EFFORT`, defaulting
to `minimal` for the same reason: the answer is capped at 900 tokens and
reasoning bills against that cap, so an unbounded budget truncates the answer.

How the field is spelled is a per-provider difference that liberdex normalises
in code: OpenRouter takes a top-level `reasoning` object, OpenAI takes
`reasoning_effort`, and Gemini 3 rejects `"medium"` through its OpenAI shim
while accepting `low` and `high`.

## Choosing models

liberdex calls an LLM in two places, and they want different models. The
planner recalls URLs from memory, streamed, on the critical path; it is
bottlenecked on output throughput and a small local model is a legitimate
choice. The answer role reads the ranked pages and states the answer; it is off
the search deadline and wants accuracy. Each is a *role*, configured separately.

A model is named `profile/model`, where the profile is an endpoint. Eighteen are
built in: `ollama`, `lmstudio`, `llamacpp`, `vllm`, `openrouter`, `openai`,
`groq`, `together`, `mistral`, `deepseek`, `xai`, `gemini`, `cohere`,
`anthropic`, and the four subscription CLIs `claude-cli`, `codex-cli`,
`gemini-cli` and `opencode-cli`. So the common cases need no configuration:

```bash
ollama pull qwen3:8b
liberdex --planner openai --planner-model ollama/qwen3:8b "…"

export OPENROUTER_API_KEY=sk-or-…
liberdex --planner openai \
    --planner-model openrouter/google/gemini-3.5-flash-lite \
    --answer-model  openrouter/anthropic/claude-sonnet-4.6 --answer synthesize "…"
```

`~/.config/liberdex/models.toml`, then a project-local `liberdex.toml`, override
the built-ins and add endpoints of your own. Python 3.12 ships `tomllib`, so
this costs no dependency:

```toml
[models.planner]
model = "ollama/qwen3:8b"
[models.answer]
model = "openrouter/anthropic/claude-sonnet-4.6"

[providers.myllm]                          # any OpenAI-compatible endpoint
base_url = "https://llm.internal.corp/v1"
api_key_env = "MYLLM_KEY"                  # the variable's *name*, never its value
headers = { "X-Tenant" = "search" }
max_tokens_param = "max_completion_tokens" # what OpenAI's reasoning models want
extra_body = { chat_template_kwargs = { enable_thinking = false } }

[providers.mycli]                          # any command that reads a prompt, prints a reply
kind = "cli"
argv = ["my-agent", "--model", "{model}", "-"]   # "{model}" substituted; prompt on stdin
default_model = "large"
```

A `kind = "cli"` provider is how a subscription you already pay for becomes the
planner or the answer model: Codex, Gemini CLI, OpenCode, or anything else with
a non-interactive mode. An empty string in `argv` marks where the
prompt goes as an argument instead of stdin (`gemini --prompt ""`). A bare
profile name (`--planner-model codex-cli`) drops the `--model` pair and the
CLI uses its own configured default, which is the one model name certain to
exist on that machine; `codex-cli/gpt-5.5` names one. The reply arrives whole,
so the plan does too: use `deep`, where the second pass makes up for the late
start, and read `liberdex models check`, which says which of these binaries is
on this machine.

Precedence is CLI flag, then environment (`LIBERDEX_PLANNER_MODEL`,
`LIBERDEX_ANSWER_MODEL`, plus the conventional `OPENAI_API_KEY`,
`OPENROUTER_API_KEY`, `GROQ_API_KEY` names), then config file, then the
built-in default. The answer role follows the planner when unset, so one
configured key runs the whole engine.

```
$ liberdex models check
planner  openrouter/google/gemini-3.5-flash-lite
answer   openrouter/google/gemini-3.5-flash-lite

  claude-cli   ok        0ms  claude_cli                      claude binary on PATH
  ollama       no        4ms  http://localhost:11434/v1       ConnectError: …
  openrouter   ok      323ms  https://openrouter.ai/api/v1    396 models
  …
```

Why no LiteLLM. Nearly every provider speaks OpenAI `/chat/completions`
natively, local runtimes included, so the streaming client is already the
model-agnostic layer. Per-profile `headers` and `extra_body` absorb the
provider quirks without a vendor adapter; only `reasoning_effort` is normalised
in code, because switching thinking off is the latency lever.

The answer layer. `answer: "extract"` returns the fact and nothing else;
`answer: "synthesize"` composes across the ranked pages with inline citations.
Either way the model reads the top eight pages rather than a snippet, meaning
their opening plus the query-selected windows, about three thousand characters
each, and its first line names the pages that were about the question. That line is
the one judgement in the pipeline made on whole pages, so it orders the SERP:
listed pages first, the pages it never saw after them, and the pages it read
and left out dropped (down to a floor of three rows).
The prompts are liberdex's own, and they lean on what this engine hands a model
that a snippet-only engine cannot: a calibrated relevance score per page, and a
`source` saying whether it came from parametric memory, a site-native route, or
a hub.
