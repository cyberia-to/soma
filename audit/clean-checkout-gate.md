---
tags: soma, audit, launch, clean-checkout
crystal-type: audit
crystal-domain: cyber
status: open
---
# soma clean-checkout gate — row 39

Property #39 — every phase-1 component builds and tests from a clean checkout
of its default branch. [[cyber/launch]] lists soma in the phase-1 dependency
closure ("the mind's tasks in cyb"); the 2026-09-22/23 sweep that found row 39
broken elsewhere (bbg, cybergraph, foculus, mudra, neuron, nox, prysm, zheng,
true-cyber, glia, honeycrisp, rune, vault, radio) did not examine soma. This
measures it directly.

## finding

`origin/main` at 7cfa6f8 has no Rust code at all: no `Cargo.toml` anywhere in
the tree, so `cargo check`/`cargo test` have nothing to run. The two crates
the rest of the graph already refers to — `soma-kernel` (`kernel/`,
depended on by nothing yet outside this repo) and `soma-agent` (`agent/`,
`path = "../agent"` from `kernel/Cargo.toml`) — exist only on the pushed
branch `origin/docs/foculus-links` (b0d5d42, 5 commits ahead of `main`, diff
`+6330/-167` across `agent/`, `kernel/`, `audit/`, `specs/`, plus a
`soma-spec.md` rewrite). That branch already has an open pull request,
[soma#1](https://github.com/cyberia-to/soma/pull/1) (titled "docs: migrate
[[sync]] to [[foculus]]" — the title describes only the doc-rename commit,
not the kernel/agent code the branch actually carries), unmerged since
2026-09-23.

This is a more severe row-39 case than the pin/path drift found in the other
14 repos: it isn't that soma fails to build clean, it's that the branch the
release train would build (`main`) carries no implementation to check at
all. The gate can't even run.

## verified

```
$ git -C /tmp/launch-work-a/soma ls-tree HEAD
100644 blob ... CLAUDE.md
100644 blob ... README.md
040000 tree ... research
100644 blob ... soma-spec.md
$ find /tmp/launch-work-a/soma -name Cargo.toml
(no output)
$ git -C /tmp/launch-work-a/soma diff --stat origin/main origin/docs/foculus-links
 agent/src/adapters.rs            |  126 +
 agent/src/controls.rs            |  456 +
 agent/src/lib.rs                 | 1000 +
 ...
 25 files changed, 6330 insertions(+), 167 deletions(-)
```

## remains

Merging soma#1 closes this row's soma slice — it is the same content this
audit measured, already reviewable. Until then, soma is phase-1's one
component where row 39 has no code-side fix available to a worker: the fix
is merging a PR that already exists, not opening a new one.
