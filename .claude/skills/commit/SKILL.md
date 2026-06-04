---
name: commit
description: Write detailed commit messages and create commits
triggers:
  - commit
  - git commit
  - write commit
  - commit message
  - create commit
---

# Commit

Use this skill to analyze staged changes and write detailed, meaningful commit messages.

## Important

- Do NOT add any bot attribution (no "Co-Authored-By" or similar)
- The commit message should read as if written by the developer

## Workflow

### Step 1: Analyze Current State

Run these commands in parallel to understand what's being committed:

```bash
# See all changes (staged and unstaged)
git status

# See staged changes (what will be committed)
git diff --cached

# See unstaged changes
git diff

# See recent commits for style reference
git log --oneline -10
```

### Step 2: Determine What to Commit

Use AskUserQuestion if there are unstaged changes:

**Question**: "There are unstaged changes. What would you like to commit?"
**Header**: "Scope"
**Options**:
1. **Staged changes only** - "Commit only what's currently staged"
2. **Stage and commit all** - "Stage all changes, then commit"
3. **Let me choose files** - "I'll specify which files to stage"

If user wants to choose files, ask:

**Question**: "Which files should be staged?"
**Header**: "Files"
(Show list of modified files as options with multiSelect: true)

### Step 3: Analyze Changes for Commit Message

Read the diff carefully and identify:

1. **Type of change**:
   - `feat`: New feature
   - `fix`: Bug fix
   - `refactor`: Code restructuring without behavior change
   - `perf`: Performance improvement
   - `test`: Adding or updating tests
   - `docs`: Documentation changes
   - `chore`: Build, config, or tooling changes
   - `style`: Formatting, whitespace (no code change)

2. **Scope**: What component/area is affected (e.g., `microbench`, `runner`, `config`, `reporting`)

3. **Summary**: What was done and why (not how)

4. **Key changes**: Bullet points of significant modifications

### Step 4: Generate Commit Message

Structure the commit message:

```
<type>(<scope>): <concise summary>

<detailed explanation of what changed and why>

Key changes:
- <change 1>
- <change 2>
- <change 3>
```

**Guidelines**:
- First line: max 72 characters, imperative mood ("Add feature" not "Added feature")
- Body: wrap at 72 characters
- Explain the "why", not just the "what"
- Reference issue numbers if applicable (e.g., "Fixes #123")

### Step 5: Present Commit Message

Show the proposed message and use AskUserQuestion:

**Question**: "How would you like to proceed?"
**Header**: "Action"
**Options**:
1. **Commit with this message** - "Create the commit as shown"
2. **Edit message** - "Let me modify the message"
3. **Regenerate** - "Generate a different message"
4. **Cancel** - "Don't commit yet"

### Step 6: Create Commit

If user approves:

```bash
git commit -m "$(cat <<'EOF'
<commit message here>
EOF
)"
```

Then verify:

```bash
git log -1
```

## Pre-Commit Checklist

Before committing, verify:

1. **Lint passes**: `make lint`
2. **Tests pass**: `make test/unit` or `make test/failed-only`
3. **No debug code**: Remove temporary print statements
4. **No secrets**: Check for hardcoded credentials or API keys

If any check fails, inform user and offer to fix before committing.
