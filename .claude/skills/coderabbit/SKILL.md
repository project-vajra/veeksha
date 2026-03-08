---
name: coderabbit
description: Run CodeRabbit CLI code review and interactively fix issues
triggers:
  - coderabbit
  - code review
  - review code
  - analyze code
  - code quality
---

# CodeRabbit Interactive Code Review

Use this skill to run CodeRabbit CLI code reviews and interactively fix reported issues.

## Important: Interactive Workflow

1. **Group similar issues together** (e.g., all unused imports, all type issues)
2. **Show code context** for each finding in the initial presentation
3. **Ask once per group** whether to fix all issues in that group
4. **Apply fixes immediately** after user approval - NO additional confirmation prompts

## Workflow

### 1. Run CodeRabbit CLI

```bash
# Default: Compare against origin/main
coderabbit review --prompt-only --base origin/main

# Alternative: For uncommitted changes only
coderabbit review --prompt-only --type uncommitted
```

**Note**: CodeRabbit reviews can take 7-30+ minutes. Inform the user.

### 2. Parse and Cluster Findings

Group findings into categories such as:
- **Unused imports** - Remove unused import statements
- **Type issues** - Fix type annotations
- **Error handling** - Add defensive checks
- **Code quality** - Constants, documentation
- **Naming** - Convention violations
- **Other** - Miscellaneous

### 3. Present Each Cluster with Analysis

For each cluster, present ALL findings with:
1. The finding details and code context
2. **Your analysis/opinion** about whether to fix it and why

**Be opinionated:**
- "CRITICAL - Must fix before merge"
- "Good catch - should be fixed"
- "Low priority - style preference, not a bug"
- "False positive - code is correct because..."

Use AskUserQuestion ONCE per cluster:
- "Yes - Fix all" - Apply all fixes
- "No - Skip all" - Skip entire cluster
- "Review individually" - One-by-one for this cluster

### 4. Apply Approved Fixes

When user approves a cluster:
1. Read each file to understand context
2. Apply fixes
3. Show a summary of changes made
4. **Immediately move to next cluster**

### 5. After All Fixes

```bash
# Verify lint passes
make lint

# Run tests
make test/failed-only

# Optional: Run CodeRabbit again to verify
coderabbit review --prompt-only --base origin/main
```

## If CodeRabbit Not Installed

```bash
curl -fsSL https://cli.coderabbit.ai/install.sh | sh
```
