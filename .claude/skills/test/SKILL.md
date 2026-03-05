---
name: test
description: Run tests intelligently and help fix failures
triggers:
  - test
  - run tests
  - run test
  - test failure
  - fix test
  - failing test
  - debug test
---

# Smart Test Runner

Use this skill to run tests intelligently, analyze failures, and help fix issues.

## Workflow

### Step 1: Determine Test Scope

Use AskUserQuestion to determine what to test:

**Question**: "What would you like to test?"
**Header**: "Test Scope"
**Options**:
1. **Failed tests only (Recommended)** - "Rerun only tests that failed in the last run. Fastest iteration."
2. **All unit tests** - "Run all unit tests"
3. **E2E tests** - "Run end-to-end tests"
4. **All tests** - "Run unit + E2E tests"
5. **Specific test file/pattern** - "Run tests matching a specific pattern"

### Step 2: Run Tests

Based on selection, run appropriate command:

#### Failed Tests Only (Fastest Iteration)
```bash
make test/failed-only
```

#### All Unit Tests
```bash
make test/unit
```

#### E2E Tests
```bash
make test/e2e
```

#### All Tests
```bash
make test
```

#### Specific Pattern
Ask user for pattern, then run:
```bash
python -m pytest tests/ -k "<pattern>" -v --tb=short
```

### Step 3: Analyze Failures

Parse test output and categorize failures:

```
## Test Results

**Summary**: X passed, Y failed, Z skipped

### Failed Tests

| Test | File | Error Type | Quick Summary |
|------|------|------------|---------------|
| test_example | tests/unit/test_example.py:45 | AssertionError | Expected 5, got 3 |
```

### Step 4: Deep Dive into Failures

For each failed test, provide detailed analysis:

```
## Failure Analysis

### 1. test_example (tests/unit/test_example.py:45)

**Error**:
```
AssertionError: assert result == expected
```

**Likely Cause**:
1. [Possible cause 1]
2. [Possible cause 2]

**Related Code**:
- Source: `veeksha/module.py:120`
```

### Step 5: Offer to Fix

Use AskUserQuestion for each failure:

**Question**: "How would you like to handle this failure?"
**Header**: "Fix Strategy"
**Options**:
1. **Investigate root cause** - "Read related code and find the actual bug"
2. **Update test expectation** - "If the new behavior is correct, update the test"
3. **Skip for now** - "Move to next failure"
4. **Show full traceback** - "Display complete error output"

### Step 6: Apply Fixes

When applying fixes:

1. Read the full test file
2. Apply the fix
3. Re-run just that test to verify:

```bash
python -m pytest tests/path/to/test.py::test_name -v --tb=short
```

4. Report result

### Step 7: Final Verification

After all fixes:

```bash
make test/failed-only
```

## Test Types Reference

| Command | Description | When to Use |
|---------|-------------|-------------|
| `make test` | All tests (unit + E2E) | Before major releases |
| `make test/unit` | Unit tests | Before commits, after changes |
| `make test/e2e` | End-to-end tests | Testing full workflows |
| `make test/failed-only` | Only failed tests | Fast iteration during fixes |

## Common Failure Patterns

### Pattern 1: Import Error
```
ModuleNotFoundError: No module named 'veeksha.xxx'
```
**Fix**: `pip install -e .` or `pip install -e ".[dev]"`

### Pattern 2: Missing Test Data
```
FileNotFoundError: [Errno 2] No such file or directory
```
**Fix**: Check test fixtures and data paths

### Pattern 3: Type Error
```
TypeError: unexpected keyword argument
```
**Fix**: Check function signatures match test expectations

## Quick Commands

```bash
# Rerun failed tests (fastest)
/test --failed

# Run all unit tests
/test --all

# Run specific test file
/test tests/unit/test_session.py

# Run tests matching pattern
/test -k "microbench"
```
