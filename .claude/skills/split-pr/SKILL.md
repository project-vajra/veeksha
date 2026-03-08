---
name: split-pr
description: Split a bloated PR into a stack of small, reviewable PRs
triggers:
  - split pr
  - split pull request
  - break up pr
  - pr stack
  - stacked prs
---

# Split PR into a Stack of Reviewable PRs

Use this skill to break a large, multi-concern PR into a stack of small, self-contained, reviewable PRs.

## Workflow

### Step 1: Analyze the Current Branch

```bash
git diff --stat origin/main..HEAD | tail -5
git diff --name-only origin/main..HEAD | sort
git log --oneline origin/main..HEAD
```

### Step 2: Group Changes by Logical Area

Read through the diff and categorize every changed file into logical groups. Common strategies:

- **By layer**: configs, core logic, CLI, tests
- **By feature**: each independent feature gets its own group
- **By dependency**: foundational changes first, dependent changes later

For each group, identify:
- What files belong to it
- What other groups it depends on
- Whether it can pass tests independently

Use `AskUserQuestion` to confirm the grouping.

### Step 3: Determine PR Stack Order

Order the PRs so that:
1. Each PR can work independently (with its dependencies merged)
2. Foundational changes (configs, types) come first
3. Feature changes come after their dependencies
4. Tests and docs come last

### Step 4: Create Branches and PRs

For each PR in the stack:

```bash
git checkout -b <branch-name> <base>

# Extract relevant changes
git checkout <original-branch> -- <file1> <file2>

# Verify lint passes
make lint

# Verify tests pass
make test/unit

# Push and create PR
git push -u origin <branch-name>
gh pr create --title "<title>" --body "$(cat <<'EOF'
## Summary
<description>

## PR Stack
This is part of a PR stack:
- PR 1: <title> (base)
- **PR 2: <this PR>** <- you are here
- PR 3: <title>

## Test plan
- [ ] `make lint` passes
- [ ] `make test/unit` passes
EOF
)"
```

### Step 5: Verify Each PR

For each created PR, verify:
1. `make lint` passes
2. `make test/unit` passes
3. The diff is small and focused (ideally < 500 lines)

### Step 6: Document the Stack

```
## PR Stack Summary

| # | PR | Title | Base | Files | Status |
|---|-----|-------|------|-------|--------|
| 1 | #XX | title | main | N | Ready |
| 2 | #YY | title | PR 1 | N | Ready |
```

## Tips

- **Mixed commits**: Use `git checkout <branch> -- <file>` to extract specific files
- **Lint verification**: Always run `make lint` after extracting changes
- **Stacked base branches**: Each PR should target the previous PR's branch
- **Test files**: Tests can go with their most relevant group or in the last PR
