# Debugging: [Brief Description]

**Date**: YYYY-MM-DD  
**Project**: [Project Name]  
**Priority**: [Critical | High | Medium | Low]  
**Status**: [Investigating | In Progress | Resolved | Blocked]

## 🐛 Problem Description

### What's Happening
Clear description of the bug or issue:
- 
- 

### Expected Behavior
What should happen:
- 
- 

### Actual Behavior
What's actually happening:
- 
- 

### Impact
- **Users affected**: XX / all users
- **Severity**: Production down / Feature broken / Minor annoyance
- **Business impact**: 

## 📋 Environment

**Environment**: [Production | Staging | Development | Local]  
**OS**: [Linux / Windows / MacOS]  
**Browser**: [Chrome 120 / Firefox 121 / etc.]  
**Version/Commit**: [v1.2.3 / commit hash]

### Reproduction Steps
1. Step one
2. Step two
3. Step three
4. Bug occurs

**Reproducibility**: [Always | Sometimes | Once]  
**First occurrence**: YYYY-MM-DD

## 🔍 Investigation

### Initial Observations
- Observation 1
- Observation 2

### Hypothesis 1
**Theory**: What might be causing this

**Evidence**:
- Supporting fact 1
- Supporting fact 2

**Test**: How to verify this hypothesis
```
Test code or steps
```

**Result**: ✅ Confirmed / ❌ Disproved / ⚠️ Inconclusive

### Hypothesis 2
**Theory**: 

**Evidence**:
- 

**Test**: 
```

```

**Result**: 

## 📊 Data & Logs

### Error Messages
```
Full error message or stack trace here
```

### Relevant Logs
```
[2025-02-03 14:23:45] ERROR: Something went wrong
[2025-02-03 14:23:45] Stack trace...
```

### Database Queries
```sql
-- Query that's causing issues
SELECT * FROM users WHERE id = 123;
```

### Network Requests
```
GET /api/users/123
Status: 500
Response: {"error": "Internal server error"}
```

## 🔧 Attempted Solutions

### Attempt 1: [Description]
**Date**: YYYY-MM-DD  
**What I tried**:
```
Code or configuration changes
```

**Result**: ❌ Didn't work / ⚠️ Partially worked / ✅ Worked

**Notes**: Why it didn't work or side effects

### Attempt 2: [Description]
**Date**: YYYY-MM-DD  
**What I tried**:
```

```

**Result**: 

**Notes**: 

## ✅ Solution

### Root Cause
Clear explanation of what was causing the bug:
- 
- 

### Fix Implemented
**Date**: YYYY-MM-DD

**Changes made**:
```
Code fix here
```

**Files modified**:
- `path/to/file1.py`
- `path/to/file2.js`

**Testing performed**:
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Verified in staging
- [ ] Verified in production

## 📝 Lessons Learned

### What Went Wrong
- Root cause analysis
- Why wasn't this caught earlier?

### Prevention
How to prevent this in the future:
- [ ] Add unit test for this case
- [ ] Add validation
- [ ] Update documentation
- [ ] Add monitoring/alerting
- [ ] Code review checklist update

### Similar Issues to Watch For
- Related areas of code that might have similar bugs
- Patterns to watch out for

## 📚 References
- [Related Documentation](link)
- [Stack Overflow Thread](link)
- [GitHub Issue](link)
- [Related Bug](../debugging/another-issue.md)

## 👥 People Consulted
- **Name**: What they suggested
- **Name**: Their insight

## ⏱️ Time Tracking
- **Time spent investigating**: X hours
- **Time spent fixing**: X hours
- **Total time**: X hours

## 🔗 Related
- [Project Notes](../projects/project-name.md)
- [Similar Issue from 2024-12](another-bug.md)

---
Tags: #debug #bug #resolved #project-name
