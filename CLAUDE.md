# CLAUDE.md

Guidance for AI coding agents (Claude Code and others) working on this repository.

Supplementary documents — read them when the task touches their area:

- [AGENTS.md](AGENTS.md): how to build, set up the runtime environment, run tests, and lint. Read it before building or running anything in this repo.
- [CONTRIBUTING.md](CONTRIBUTING.md): general contribution rules (DCO sign-off, coding style, review process).

## Branch and PR policy

- **Never create or push branches to the upstream repository (`nnstreamer/nnstreamer`), even if the account you operate under has write access.** Maintainers have branch-push permission on upstream; do not use it for work branches.
- All work branches MUST be created on the contributor's **personal fork** (e.g., `github.com/<user>/nnstreamer`).
- Pull requests MUST be opened **from the fork branch** against `nnstreamer/nnstreamer:main`.
- Before any `git push`, verify the target remote points to the personal fork (`git remote -v`), not to `nnstreamer/nnstreamer`.
- Do not merge PRs; merging is done manually by maintainers.

## Test case policy

Every functional change must come with test cases in the same PR:

- **New feature**: add test cases that cover the new behavior — both positive cases and negative (failure/error-path) cases.
- **Bug fix**: add a regression test that reproduces the bug (fails without the fix, passes with it). A fix without a reproducing test is incomplete unless the bug is genuinely untestable (e.g., build-script-only changes); state the reason in the PR body if so.
- Choose the appropriate test layer:
  - **GTest** unit tests under `tests/` (per-component directories, e.g., `tests/nnstreamer_filter_*`, `tests/common`) for API/function-level testing. Run with `meson test -C build/ -v`.
  - **SSAT** pipeline tests (`runTest.sh` scripts under `tests/`) for gst-launch-based golden testing. Run with `cd tests && ssat`. Negative SSAT cases use ID suffix `_n` and expect failure: `gstTest "..." <id>_n 0 1 $PERFORMANCE`.
- Test-writing guides: [Documentation/how-to-use-testcases.md](Documentation/how-to-use-testcases.md) and [Documentation/how-to-write-testcase.md](Documentation/how-to-write-testcase.md).
- Run the relevant test suites locally (or confirm CI passes) before marking a PR ready for review.

## Commit conventions

- Subject line: `[Component] Summary` (e.g., `[Filter/TensorRT10] Fix memory leak on unload`).
- Body explains what and why; wrap at reasonable width.
- Every commit requires a DCO sign-off line: `Signed-off-by: Name <email>`. Do not duplicate the sign-off.
- Every function (including static functions and test cases) needs a Doxygen comment; CI enforces this.

## Code style

- C: K&R style, 2-space indentation (checked by CI; see `.github/workflows/static.check.yml`).
- C++: repo `.clang-format`.
- Match the style of surrounding code; do not reformat unrelated lines.
