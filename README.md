# Engineering Notes System for Notepad++

A structured approach to technical note-taking for software engineers and AI engineers.

## 📁 Folder Structure

```
notes/
├── daily/           # Daily work logs (YYYY-MM-DD.md)
├── projects/        # One file per project
├── learning/        # Courses, tutorials, concepts
├── snippets/        # Reusable code snippets by language
├── debugging/       # Problem-solving notes and solutions
├── ideas/           # Random thoughts, TODOs, future projects
└── templates/       # Template files (this folder)
```

## 🚀 Quick Start

1. Copy this entire folder to your preferred location (e.g., `C:\dev\notes\` or `~/Documents/notes/`)
2. Open Notepad++ and add this folder to your workspace
3. Copy templates from `templates/` folder to create new notes
4. Start with a daily note using `templates/daily-template.md`

## 📝 Daily Workflow

### Morning
1. Create a new daily note: `daily/2025-02-03.md`
2. Copy content from `templates/daily-template.md`
3. Fill in your goals for the day

### Throughout the Day
- Keep your daily note open in the first tab
- Log what you're working on with timestamps
- Quick captures in `scratch.txt` (keep always open)
- Add code snippets to appropriate snippet files

### End of Day
- Review accomplishments
- Note any blockers or questions
- Plan tomorrow's tasks
- Move important items from `scratch.txt` to proper files

## 💡 Tips

### Search Everything
- **Ctrl+Shift+F**: Search across all note files
- Use consistent tags: `#bug`, `#todo`, `#important`, `#review`

### Bookmarks
- **Ctrl+F2**: Toggle bookmark on important lines
- **F2**: Jump to next bookmark

### Multi-Tab Workflow
Keep these tabs always open:
1. Today's daily note
2. Current project note
3. scratch.txt (quick capture)
4. snippets file for your current language

### Naming Conventions
- Daily notes: `YYYY-MM-DD.md` (e.g., `2025-02-03.md`)
- Project notes: `project-name.md` (lowercase, hyphens)
- Learning notes: `topic-name.md`

### Tags for Quick Searching
Use consistent tags in your notes:
- `#todo` - Action items
- `#bug` - Known issues
- `#idea` - Future features or projects
- `#learn` - Topics to study
- `#question` - Things to research
- `#important` - Critical information
- `#review` - Code or notes to review later

### Weekly Review
Every Friday or Monday:
1. Review all daily notes from the past week
2. Consolidate learnings into project or learning notes
3. Clear completed TODOs
4. Archive or clean up scratch.txt

## 🔌 Recommended Notepad++ Plugins

Install via Plugins → Plugins Admin:

1. **NppMarkdownPanel** - Live Markdown preview
2. **Explorer** - Better file navigation sidebar
3. **Compare** - Diff two versions of notes
4. **NppExec** - Run scripts on your notes

## ⚙️ Notepad++ Settings

### Enable Markdown Highlighting
Settings → Style Configurator → Language: Markdown

### Show Function List
View → Function List (shows Markdown headers as outline)

### Auto-backup
Settings → Preferences → Backup
- Enable "Remember current session for next launch"
- Enable backup on save

## 🎯 Advanced Techniques

### Linking Between Notes
Use relative paths in your notes:
```markdown
See also: [Project XYZ Notes](../projects/xyz.md)
Reference: [Auth Debugging](../debugging/auth-issue-2025-01.md)
```

### Code Blocks with Language
Always specify language for syntax highlighting:
```python
def hello_world():
    print("Hello, World!")
```

### Quick Templates with Macros
Record macros for repetitive tasks:
1. Macro → Start Recording
2. Type your template
3. Macro → Stop Recording
4. Macro → Save Current Recorded Macro

### Git Integration (Optional)
Initialize git in your notes folder to:
- Version control your notes
- Sync across devices
- Track your learning journey

```bash
cd /path/to/notes
git init
git add .
git commit -m "Initial notes"
```

## 📊 Tracking Progress

### Weekly Summary Template
Create a `weekly-summary-YYYY-WW.md` file:
```markdown
# Week 5 - 2025

## Accomplished
- 

## Challenges
- 

## Learnings
- 

## Next Week Goals
- 
```

## 🔍 Search Examples

Find all TODOs across all notes:
```
Ctrl+Shift+F
Find what: #todo
Directory: C:\dev\notes
Filter: *.md
```

Find all Python snippets:
```
Find what: ```python
Directory: C:\dev\notes\snippets
```

## 📚 Resources

- [Markdown Guide](https://www.markdownguide.org/)
- [Notepad++ Documentation](https://npp-user-manual.org/)
- [Zettelkasten Method](https://zettelkasten.de/introduction/) - For linking notes

---

**Remember**: The best note-taking system is the one you actually use. Start simple and evolve your system as needed.
