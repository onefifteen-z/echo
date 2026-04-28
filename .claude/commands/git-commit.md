---
name: git-commit
description: Stage all changes (git add -A) and create a commit with a generated message. Use when the user wants to save work to git with a clear summary.
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
   - Summarizes the staged change in **one** subject line (imperative mood, ~72 characters or less).
   - Optionally adds a blank line and a short body only if multiple distinct concerns need explaining.
   - Uses **English** for the message (match repository conventions; this project uses English for commits and docs).
   - Incorporates any extra intent from: $ARGUMENTS
3. Run exactly:

```bash
git commit -m "<subject line>" # use -m twice if you add a body: git commit -m "subject" -m "body"
```

4. Confirm the new commit hash and one-line summary from `git log -1 --oneline`.

**Do not** amend, force-push, or change remote branches unless the user explicitly asks.
