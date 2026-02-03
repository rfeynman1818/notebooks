# Quick Reference Guide

## 📁 File Organization

```
notes/
├── daily/              # Daily logs (YYYY-MM-DD.md)
├── projects/           # One file per project
├── learning/           # Courses, tutorials, concepts
├── snippets/           # Code snippets by language
├── debugging/          # Bug fixes and troubleshooting
├── ideas/              # Brainstorming and future projects
├── templates/          # Template files
└── scratch.txt         # Quick capture space
```

## 🎯 Daily Workflow

### Morning (5 minutes)
1. Copy `templates/daily-template.md` to `daily/YYYY-MM-DD.md`
2. Fill in today's goals
3. Open in first tab

### During the Day
- Log work with timestamps
- Quick notes in `scratch.txt`
- Add snippets to appropriate files

### End of Day (5 minutes)
- Review accomplishments
- Note blockers
- Plan tomorrow
- Organize `scratch.txt`

## ⌨️ Essential Notepad++ Shortcuts

### Navigation
- `Ctrl+Tab` - Switch between tabs
- `Ctrl+W` - Close current tab
- `Ctrl+Shift+T` - Reopen closed tab
- `Ctrl+G` - Go to line number
- `F2` / `Shift+F2` - Next/previous bookmark

### Search
- `Ctrl+F` - Find in current file
- `Ctrl+H` - Find and replace
- `Ctrl+Shift+F` - Find in files (search all notes!)
- `Ctrl+F3` - Find next occurrence

### Bookmarks
- `Ctrl+F2` - Toggle bookmark
- `F2` - Go to next bookmark
- `Shift+F2` - Go to previous bookmark

### Editing
- `Ctrl+D` - Duplicate line
- `Ctrl+L` - Delete line
- `Ctrl+Shift+Up/Down` - Move line up/down
- `Ctrl+Q` - Block comment
- `Tab` / `Shift+Tab` - Indent/unindent

### Multi-cursor
- `Ctrl+Alt+Up/Down` - Add cursor above/below
- Hold `Alt` and drag - Column selection

## 🏷️ Tagging System

Use these tags at the end of your notes for easy searching:

### Status Tags
- `#todo` - Action items
- `#in-progress` - Currently working on
- `#blocked` - Waiting on something
- `#completed` - Finished tasks

### Priority Tags
- `#urgent` - High priority
- `#important` - Medium priority
- `#someday` - Low priority / future

### Category Tags
- `#bug` - Bug-related
- `#feature` - New feature
- `#idea` - Brainstorming
- `#learning` - Educational content
- `#review` - Needs review

### Project Tags
- `#project-name` - Link to specific project
- `#client-name` - Client work

### Type Tags
- `#python` `#javascript` `#react` - Technology tags
- `#api` `#database` `#frontend` - Component tags

## 🔍 Powerful Search Patterns

### Find All TODOs
```
Ctrl+Shift+F
Find what: #todo
Directory: C:\path\to\notes
Filter: *.md
Find All in All Opened Documents
```

### Find Python Snippets
```
Find what: ```python
Directory: notes\snippets
```

### Find Recent Work
```
Find what: 2025-02-\d+
Regular expression: ✓
Directory: notes\daily
```

### Find Bugs from Specific Project
```
Find what: #bug #project-name
Directory: notes
```

## 📝 Quick Templates

### Add Timestamp
Type current time: `14:23 - `
(Set up a macro for this!)

### Quick TODO
```
- [ ] Task description #todo
```

### Quick Note
```
[14:23] Brief note about something important
```

### Code Block
```language
code here
```

### Link to Another File
```markdown
[Link text](../folder/file.md)
```

## 🎨 Markdown Quick Reference

### Headers
```markdown
# H1
## H2
### H3
```

### Emphasis
```markdown
**bold**
*italic*
~~strikethrough~~
`code`
```

### Lists
```markdown
- Bullet point
- Another point

1. Numbered item
2. Another item

- [ ] Checkbox
- [x] Completed checkbox
```

### Code Blocks
```markdown
```python
def function():
    pass
```
```

### Tables
```markdown
| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
```

### Links
```markdown
[Link text](https://example.com)
[Local file](../path/to/file.md)
```

## 🔧 Notepad++ Setup Tips

### Install These Plugins
1. **NppMarkdownPanel** - Preview Markdown
2. **Explorer** - Better file browser
3. **Compare** - Diff files
4. **NppExec** - Run scripts

### Recommended Settings
1. Settings → Preferences → Backup
   - Enable session snapshot
   - Backup on save

2. Settings → Preferences → Multi-Instance
   - Open session in new instance: No

3. View → Function List
   - Shows Markdown headers as outline

### Create Useful Macros

**Timestamp Macro**:
1. Macro → Start Recording
2. Type: `HH:MM - `
3. Macro → Stop Recording
4. Save macro with shortcut (e.g., `Ctrl+Shift+T`)

## 💡 Pro Tips

### Multi-File Editing
Keep these tabs always open:
1. Today's daily note
2. Current project note
3. scratch.txt
4. Relevant snippet file

### Weekly Review
Every Friday:
1. Search: `#todo` in daily notes
2. Move incomplete TODOs to next week
3. Archive completed items
4. Clean up scratch.txt

### Link Your Notes
Create connections:
```markdown
See also: [Related Note](../projects/project.md)
References: [Bug Fix](../debugging/issue.md)
```

### Use Consistent Naming
- Daily: `YYYY-MM-DD.md`
- Projects: `project-name.md` (lowercase, hyphens)
- Learning: `topic-name.md`

### Search Like a Pro
Combine tags:
```
#bug #urgent #project-name
```

Use regex for dates:
```
202[3-5]-\d{2}-\d{2}
```

### Version Control Your Notes
```bash
cd notes/
git init
git add .
git commit -m "Notes snapshot"
```

## 🎯 Common Use Cases

### Starting a New Project
1. Copy `templates/project-template.md` to `projects/project-name.md`
2. Fill in overview and goals
3. Add tag at bottom: `#project #active`

### Debugging an Issue
1. Copy `templates/debugging-template.md` to `debugging/issue-name.md`
2. Document the problem
3. Log investigation steps
4. Document solution

### Learning Something New
1. Copy `templates/learning-template.md` to `learning/topic-name.md`
2. Take notes as you learn
3. Add code examples
4. Create practice projects

### Capturing Quick Ideas
1. Write in `scratch.txt`
2. Weekly review: move to `ideas/` folder
3. Use `templates/idea-template.md` for serious ideas

## 📊 Tracking Your Progress

### Daily Metrics
Count completed tasks:
```
Search: - \[x\]
In: daily/2025-02-*.md
```

### Weekly Summary
Every Friday, create:
```
weekly-summary-YYYY-WW.md
```

### Monthly Review
Search pattern:
```
#monthly-review #YYYY-MM
```

## 🚀 Level Up

### Advanced Organization
Create index files:
```markdown
# Projects Index

## Active Projects
- [Project A](projects/project-a.md) - Status: In progress
- [Project B](projects/project-b.md) - Status: Planning

## Completed Projects
- [Project C](projects/project-c.md) - Completed: 2024-12
```

### Knowledge Base
Build a personal wiki:
- Create topic pages in `learning/`
- Link related concepts
- Add code examples
- Reference from daily notes

### Automation Ideas
1. Script to create daily notes automatically
2. Git commit notes every evening
3. Generate weekly summaries from daily notes
4. Search script to find specific patterns

---
**Remember**: The best system is the one you actually use consistently.
Start simple, then add complexity as needed!
