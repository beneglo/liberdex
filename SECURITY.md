# Security

liberdex fetches pages from the open web on behalf of whoever runs it, so
a bug in fetching, extraction or the HTTP server can matter beyond the
machine it runs on. Reports are welcome.

## Reporting

Use GitHub's private vulnerability reporting: the "Report a vulnerability"
button under the Security tab of this repository. It opens a private
advisory that only the maintainer can read. Please do not open a public
issue for anything you believe is exploitable.

Include what you can: the version or commit, how liberdex was started
(`liberdex serve`, `liberdex mcp`, the container), the request or page that
triggers it, and what you observed.

You should hear back within seven days. Fixes ship as a normal release and
the advisory is published once it is out.

## Scope

In scope: the engine and its HTTP and MCP servers, the settings drawer, the
install helpers for host apps, and the container image.

Out of scope: content of third-party pages liberdex fetches, rate limits or
blocks imposed by the sites it visits, and the behaviour of the language
model that writes the plan.

## Supported versions

Only the latest release receives fixes.
