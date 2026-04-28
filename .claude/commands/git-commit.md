---
name: git-commit
description: Stage all changes (git add -A) and create a commit with a typed conventional message (feat/fix/chore/…).
disable-model-invocation: true
allowed-tools: Bash(git add *) Bash(git commit *) Bash(git status *) Bash(git diff *) Bash(git log *)
argument-hint: "[optional hint for the commit message]"
---

## Staged snapshot

```!
git add -A && git status --short && echo "--- CACHED DIFF ---" && git diff --cached
```

## Your task

1. If there is nothing to commit (no staged changes after the snapshot above), say so clearly and **do not** run `git commit`.
2. Otherwise, write a **single** commit message that:
   - **Subject line (required format):** `<type>: <description>`
     - **`<type>`** must be exactly one of, lowercase: `feat`, `fix`, `bug`, `chore`, `docs`, `refactor`, `test`, `perf`, `style`, `ci`, `build`, `revert`.
     - Use **`feat:`** for new behavior or UX. Use **`fix:`** or **`bug:`** for defect repairs (pick one per repo habit; default to **`fix:`** when unsure). Use **`chore:`** for tooling, deps, or housekeeping with no user-facing change.
     - Then **a colon and a single space** (`: `), then a short **imperative** description (no trailing period), **~72 characters or less for the whole subject line**.
   - Optionally adds a blank line and a short body only if multiple distinct concerns need explaining.
   - Uses **English** for the message (match repository conventions; this project uses English for commits and docs).
   - Incorporates any extra intent from: $ARGUMENTS
3. Run exactly:

```bash
git commit -m "<subject line>" # use -m twice if you add a body: git commit -m "subject" -m "body"
```

4. Confirm the new commit hash and one-line summary from `git log -1 --oneline`.

**Do not** amend, force-push, or change remote branches unless the user explicitly asks.
