---
name: lint
description: Run formatters and linters, then fix reported issues
triggers:
  - lint
  - format
  - format and lint
  - check style
  - fix style
  - code style
  - fix formatting
  - fix linting
---

# Lint and Format

Use this skill to run code formatters and linters, then interactively fix any issues found.

## Workflow

### Step 1: Ask User What to Run

Use AskUserQuestion to determine scope:

**Question**: "What would you like to do?"
**Header**: "Action"
**Options**:
1. **Format & Lint all** - "Run all formatters then all linters (recommended before commits)"
2. **Format only** - "Run formatters to auto-fix style issues"
3. **Lint only** - "Run linters to check for issues without auto-fixing"
4. **Specific check** - "Choose a specific linter or formatter"

### Step 2A: Format & Lint All

Run formatters first (they auto-fix), then linters (they report issues):

```bash
# Format all code (auto-fixes style)
make format

# Then lint to catch remaining issues
make lint
```

### Step 2B: Format Only

```bash
make format
```

This runs all formatters:
- `isort` - Python import ordering
- `autoflake` - Remove unused Python imports
- `black` - Python code style

### Step 2C: Lint Only

```bash
make lint
```

This runs all linters:
- `isort --check` - Python import order check
- `black --check` - Python style check
- `autoflake --check` - Unused Python imports check
- `codespell` - Spelling check
- `pyright` - Python type checking

### Step 2D: Specific Check

Use AskUserQuestion:

**Question**: "Which specific checks would you like to run?"
**Header**: "Check Type"
**multiSelect**: true
**Options**:
1. **Python formatting** - "black, isort, autoflake"
2. **Python type checking** - "pyright"
3. **Spelling** - "codespell"

Run selected checks:

```bash
# Python formatting
make format/isort && make format/autoflake && make format/black

# Python type checking
make lint/pyright

# Spelling
make lint/codespell
```

### Step 3: Analyze Results

Parse the output and categorize issues:

```
## Results Summary

| Check | Status | Issues |
|-------|--------|--------|
| black | [Pass/Fail] | N files reformatted |
| isort | [Pass/Fail] | N import issues |
| autoflake | [Pass/Fail] | N unused imports |
| codespell | [Pass/Fail] | N spelling errors |
| pyright | [Pass/Fail] | N type errors |
```

### Step 4: Fix Issues Interactively

If linting found issues that can't be auto-fixed:

#### For Pyright Type Errors

Group by error type and present:

```
## Type Errors (pyright)

### Missing Type Annotations (5 issues)

| File | Line | Issue |
|------|------|-------|
| veeksha/core/engine.py | 45 | Parameter "config" has no type annotation |

Would you like me to fix these by adding type annotations?
```

Use AskUserQuestion:
- "Fix all type annotations" - Add missing type hints
- "Fix one by one" - Review each individually
- "Skip" - Move to next category

#### For Spelling Errors (codespell)

```
## Spelling Errors

| File | Line | Found | Suggested |
|------|------|-------|-----------|
| veeksha/utils/helpers.py | 12 | "recieve" | "receive" |

Would you like me to fix these spelling errors?
```

### Step 5: Apply Fixes

When fixing issues:

1. Read each file
2. Apply the fix
3. Show before/after for each change
4. Re-run the specific linter to verify fix

```bash
# After fixes, verify
make lint/pyright  # or whichever linter had issues
```

### Step 6: Final Verification

After all fixes:

```bash
# Run full lint to ensure everything passes
make lint
```

Report final status.

## Quick Commands

```bash
# Quick format (no questions, just format everything)
/lint --format-only

# Quick lint check (no fixes, just report)
/lint --check-only

# Pre-commit check (format then lint)
/lint --pre-commit
```

## Pre-Commit Workflow

When user specifies `--pre-commit` or asks to prepare for commit:

```bash
# 1. Format everything
make format

# 2. Lint everything
make lint

# 3. If lint passes, show status
git status
```

If lint fails, offer to fix issues before proceeding.
