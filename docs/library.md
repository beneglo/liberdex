# The command line and the library

## `liberdex search`

```bash
liberdex "how does the TCP three way handshake work"
liberdex search --json --debug "postgres create index concurrently"
liberdex search --planner none "what is the capital of australia"

# only fetch these sites, render the pages as markdown, cap the whole SERP
liberdex search --sites db.example --format markdown \
    --token-budget 4000 "create index concurrently locking"

# answer the question from the ranked pages. The answer model reads the top
# pages in full and says which were about the question; the SERP is reordered
# by that, and pages it read and rejected are dropped.
liberdex search --answer extract "melting point of tungsten"
liberdex search --answer synthesize "kafka vs rabbitmq durability"

# a plan written by someone else (JSON, fields in skills/liberdex/references/plan-protocol.md):
# no planner is called
liberdex search --plan-file plan.json "postgres create index concurrently locking"

# which planner will run, and which of the eighteen profiles this machine can reach
liberdex models check
```

`liberdex <query>` is `liberdex search <query>`. `--planner auto` (default)
takes the configured model, else a key in the environment, else the `claude`
binary; `--planner-model profile/model` names one outright.

## Python

```python
import asyncio
from liberdex import Liberdex
from liberdex.planner import OpenAIPlanner

async def main():
    async with Liberdex(planner=OpenAIPlanner()) as lx:
        lx.warmup()
        resp = await lx.search("how do mRNA vaccines work", top_k=10, budget=6.0)
        for r in resp.results:
            print(r.score, r.url, r.passages[0][:90])

asyncio.run(main())
```

With a plan of your own, which is the engine's fastest path and what the MCP
server does with the plan an agent writes:

```python
from liberdex.planner import plan_from_dict

plan = plan_from_dict("tcp three way handshake", {
    "candidates": [{"url": "https://en.encyclopedia.example/wiki/Handshake_(computing)", "prior": 0.9}],
    "routes": {"wikipedia": "TCP handshake"},
})
resp = await lx.search("tcp three way handshake", plan=plan)
```

`Liberdex(planner=...)` takes anything with the `Planner` protocol
(`liberdex/planner/base.py`): `OpenAIPlanner`, `ClaudeCLIPlanner`,
`CLIPlanner`, `SuppliedPlanner`, `cache.CachedPlanner` around any of them, or
`None` for site-native routes only. `planner.factory.make_planner("auto")`
resolves one the way the CLI and the server do.

Importing `liberdex` into a program makes that program a derivative work under
the AGPL; calling a server does not. [deploy.md](deploy.md) has the client.
