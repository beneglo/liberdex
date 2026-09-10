# Contributing to liberdex

Pull requests are welcome. Before your first one is merged you must sign the
[Contributor License Agreement](CLA.md). A bot asks for the signature on the
pull request; reply with the sentence it quotes and it records you. You sign
once and it covers every later contribution.

Why a CLA: liberdex is AGPL-3.0-or-later and the Owner also offers commercial
licences for proprietary embedding. That is only possible if every contribution
is licensable under both. The CLA is a licence grant, not a copyright
assignment. You keep ownership of your work and can use it anywhere else.

## Ground rules

- Run `uv run ruff check` and `uv run pytest` before opening a pull request;
  CI runs both. Lint is enforced, `ruff format` is not: the code is wrapped by
  hand and reads that way on purpose.
- Benchmark captures contain full page bodies from third-party sites. Never
  commit them; they stay local, and so does any saved corpus.
- Dependencies must be permissive (MIT, BSD, Apache-2.0, ISC) or file-level
  copyleft like MPL-2.0; nothing GPL or AGPL, which would rule out the
  commercial licence. Note the licence in the pull request.
- Scoring changes must come with a benchmark run, planner mode held constant,
  and negatives reported as well as wins. The harness talks
  to the live web and to a paid grader, so it is not installed with the engine
  and does not run in CI.
- Names on the API and in the docs come from liberdex's own vocabulary (plan,
  candidate, route, hub, page, passage) and never from another provider's.
- The README is the repository's front page and `site/` is liberdex.net;
  long-form material goes under `docs/`. The site is plain HTML with no build
  step, and its demo replays recorded searches on `.example` hosts rather
  than fetching anything, since a real result page belongs to whoever
  published it.
