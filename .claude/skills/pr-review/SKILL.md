---
name: pr-review
description: Comprehensive PR review with detailed explanations and constructive feedback
triggers:
  - pr-review
  - pr review
  - review pr
  - review pull request
  - detailed pr review
---

# Comprehensive PR Review

Use this skill to perform thorough, educational PR reviews with detailed explanations and constructive feedback.

## Review Philosophy

**Goal**: Provide detailed, constructive feedback that improves code quality and helps engineers grow.

**Approach**:
- Explain the reasoning behind every suggestion
- Highlight what was done well (positive reinforcement)
- Connect feedback to broader engineering principles
- Be thorough but constructive - assume good intent

## Workflow

### Step 1: Fetch PR Information

Get the PR number from the user's command. If not provided, ask for it.

**Option A: Using GitHub CLI (preferred)**
```bash
gh pr view <PR_NUMBER> --json title,body,author,baseRefName,headRefName,additions,deletions,changedFiles,commits,files
gh pr diff <PR_NUMBER>
gh pr view <PR_NUMBER> --json comments,reviews
```

**Option B: Using git diff (fallback)**
```bash
git fetch origin main
git diff origin/main...HEAD --stat
git log origin/main..HEAD --oneline
git diff origin/main...HEAD
```

### Step 2: Understand the Context

Present a brief summary:

```
## PR Overview

**Title**: [PR title]
**Author**: [author]
**Type**: [Bug Fix | Feature | Refactor | Docs | Test | Other]
**Scope**: [N files changed, +X/-Y lines]

**Summary of Intent**:
[1-2 sentences explaining what this PR is trying to do]
```

### Step 3: Analyze Each Changed File

For EVERY file in the PR, provide detailed analysis. Read each file to understand the full context (not just the diff).

### Step 4: Present Comprehensive Review

---

## PR Review: [PR Title]

### Summary of Changes

For each file changed, explain:
1. **What changed** - Describe the modifications
2. **Why it matters** - How this change relates to the PR's goal
3. **Key code snippets** - Show the most important changes with explanation

---

### Evaluation Criteria

Rate each criterion with: Pass | Needs Work | Fail

---

#### 1. Correctness

**Rating**: [Pass | Needs Work | Fail]

**What We're Checking**:
- Does the code do what it's supposed to do?
- Are there logic errors or edge cases missed?
- Will it work correctly in production?

| Location | Issue | Explanation | Suggested Fix |
|----------|-------|-------------|---------------|

---

#### 2. Code Quality

**Rating**: [Pass | Needs Work | Fail]

**What We're Checking**:
- Is the code readable and maintainable?
- Are functions focused and well-named?
- Is complexity kept to a minimum?

---

#### 3. Code Style & Standards

**Rating**: [Pass | Needs Work | Fail]

**What We're Checking**:
- Consistent naming conventions (snake_case for functions/variables)
- Proper import ordering (stdlib -> third-party -> local)
- Type hints present
- No unused imports

---

#### 4. Robustness

**Rating**: [Pass | Needs Work | Fail]

**What We're Checking**:
- Error handling for external inputs
- Edge cases handled
- Thread safety considerations (if applicable)

---

#### 5. Testing

**Rating**: [Pass | Needs Work | Fail]

**What We're Checking**:
- Are there tests for new functionality?
- Do tests cover edge cases?
- Are tests actually testing the right things?

---

#### 6. Completeness

**Rating**: [Pass | Needs Work | Fail]

**What We're Checking**:
- Does the PR fully implement the intended feature/fix?
- Are all necessary files updated?
- Are there any TODO comments that should be addressed?

---

### Overall Assessment

**Verdict**: [Approve | Request Changes | Needs Discussion]

**Strengths**:
1. [What the author did well]

**Priority Fixes Required** (must fix before merge):
1. [Critical issue]

**Suggested Improvements** (nice to have):
1. [Non-blocking suggestion]

---

### Step 5: Offer to Help Fix Issues

Use AskUserQuestion with options:
- "Fix critical issues" - Apply fixes for must-fix items
- "Walkthrough fixes" - Explain how to fix each issue step-by-step
- "Explain feedback" - Provide more context on specific points
- "Done reviewing" - End the review session

## Commands

```bash
# Review a specific PR
/pr-review 123

# Review PR with extra focus
/pr-review 123 --focus testing
/pr-review 123 --focus performance
```
