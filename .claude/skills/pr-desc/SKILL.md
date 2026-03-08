---
name: pr-desc
description: Generate PR title and description from branch diff against origin/main
triggers:
  - pr-desc
  - pr description
  - write pr
  - pr title
  - pull request description
---

# PR Description Generator

Use this skill to analyze the diff between your branch and origin/main and generate a well-structured PR title and description.

## Important

- Do NOT add any bot attribution
- The description should read as if written by the developer
- Focus on the "why" and impact, not just the "what"

## Workflow

### Step 1: Gather Branch Information

Run these commands in parallel:

```bash
# Fetch latest main
git fetch origin main

# Get diff stats
git diff origin/main...HEAD --stat

# Get full diff
git diff origin/main...HEAD

# Get commit history
git log origin/main..HEAD --oneline

# Get current branch name
git branch --show-current

# Get detailed commit messages
git log origin/main..HEAD --format="%s%n%n%b%n---"
```

### Step 2: Analyze the Changes

Group changes by category:

1. **New Files**: Files added in this branch
2. **Modified Files**: Existing files that were changed
3. **Deleted Files**: Files removed in this branch

### Step 3: Determine PR Type

- **Feature**: New functionality
- **Bug Fix**: Fixing incorrect behavior
- **Refactor**: Code restructuring without behavior change
- **Performance**: Optimization improvements
- **Test**: Adding or improving tests
- **Docs**: Documentation updates
- **Chore**: Build, config, or tooling changes

### Step 4: Generate PR Title

Format: `[<scope>] <type>: <concise description>`

**Guidelines**:
- Max 72 characters
- Imperative mood ("Add feature" not "Added feature")
- Be specific about what changed

**Examples**:
- `[Microbench] feat: Add decode window validation and safety multiplier`
- `[Runner] fix: Prevent race condition in session tracking`
- `[Config] refactor: Consolidate benchmark config hierarchy`

### Step 5: Generate PR Description

Use this structure:

```markdown
## Summary

<2-3 sentences explaining what this PR does and why>

## Changes

### <Category 1>
- <Change description with file reference>

### <Category 2>
- <Change description>

## Key Design Decisions

- <Decision 1 and rationale>

## Testing

- [ ] <How was this tested?>
- [ ] <What scenarios were verified?>

## Related Issues

Fixes #<issue_number> (if applicable)
```

### Step 6: Present and Offer Actions

Use AskUserQuestion:

**Question**: "How would you like to proceed?"
**Header**: "Action"
**Options**:
1. **Copy to clipboard** - "I'll paste this into my PR"
2. **Create PR now** - "Create the PR using gh CLI"
3. **Edit description** - "Let me modify the description"
4. **Regenerate** - "Generate a different description"

### Step 7: Create PR (if requested)

```bash
git push -u origin $(git branch --show-current)

gh pr create --title "<title>" --body "$(cat <<'EOF'
<description>
EOF
)"
```

## Quick Commands

```bash
# Generate PR description
/pr-desc

# Generate and create PR immediately
/pr-desc --create

# Generate for specific base branch
/pr-desc --base develop
```
