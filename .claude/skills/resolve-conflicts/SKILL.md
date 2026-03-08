---
name: resolve-conflicts
description: Analyze and resolve git merge conflicts with guided assistance
triggers:
  - resolve conflicts
  - fix conflicts
  - merge conflict
  - resolve merge
  - conflict resolution
  - git conflict
---

# Resolve Merge Conflicts

Use this skill to analyze git merge conflicts, understand what each side intended, and resolve them interactively.

## Important: Interactive Workflow

1. **Analyze all conflicts** - Understand what's conflicting and why
2. **Group by category** - Similar conflicts together
3. **Show context and opinion** - For each group, explain what each side did and recommend a resolution
4. **Ask once per group** - User decides how to resolve the category
5. **Apply fixes immediately** - No additional confirmation after user approves

## Workflow

### Step 1: Detect Conflict State

```bash
git status
git diff --name-only --diff-filter=U
git log --oneline -1 HEAD
git log --oneline -1 MERGE_HEAD 2>/dev/null || git log --oneline -1 REBASE_HEAD 2>/dev/null
```

Present overview:

```
## Merge Conflict Overview

**Operation**: Merging `feature/xxx` into `main`
**Conflicted Files**: N files

| File | Conflict Regions | Complexity |
|------|------------------|------------|
| path/to/file.py | 3 | Medium |
```

### Step 2: Analyze Each Conflict

Read each conflicted file and parse conflict markers. Get context from both sides:

```bash
git log --oneline main -- <file>
git log --oneline feature/xxx -- <file>
git show :1:<file> 2>/dev/null  # base version
```

### Step 3: Categorize Conflicts

1. **Import conflicts** - Different imports added at same location
2. **Function signature changes** - Both sides modified function parameters
3. **Implementation conflicts** - Different logic changes to same code
4. **Configuration conflicts** - Version bumps, dependencies, settings
5. **Test conflicts** - Test additions or modifications

### Step 4: Present Each Category with Analysis

For each category, present ALL conflicts with your analysis and recommendation.

Use AskUserQuestion ONCE per category:

**Options**:
1. **Use suggested resolution for all** - "Apply the recommended resolution"
2. **Keep ours for all** - "Use current branch only"
3. **Keep theirs for all** - "Use incoming branch only"
4. **Review individually** - "Decide each conflict separately"

### Step 5: Apply Resolutions

1. Read the full file
2. Apply the resolution (remove conflict markers, insert resolved code)
3. Stage the file: `git add <resolved_file>`
4. Move to next category

### Step 6: Final Verification

```bash
git diff --name-only --diff-filter=U
git diff --cached --stat
make lint
make test/failed-only
```

Present summary and offer to complete or abort the merge.

## Conflict Resolution Strategies

| Strategy | When to Use |
|----------|-------------|
| **Keep Ours** | Their changes are outdated or superseded |
| **Keep Theirs** | Our changes are outdated or superseded |
| **Merge Both** | Changes are independent and compatible |
| **Custom** | Changes conflict in intent, need manual reconciliation |

## Quick Commands

```bash
/resolve-conflicts
/resolve-conflicts --abort
/resolve-conflicts --status
```
