# Monthly Maintenance Checklist

**Purpose:** Keep the Knowledge Extraction Pipeline secure, up-to-date, and running smoothly

**Time Required:** ~30-45 minutes per month

**Last Updated:** 2025-11-25

---

## 🔒 Security Maintenance (Monthly)

### 1. Update Dependencies

**Why:** Security patches and bug fixes are released regularly

```bash
cd "/Users/IRI/Knowledge Base/Pipeline/Open Source/knowledge-extraction-pipeline"

# Check for outdated packages
pip list --outdated

# Update specific packages (review changes first)
pip install --upgrade chromadb sentence-transformers networkx

# Regenerate requirements if needed
pip freeze > requirements.txt
```

**Action Items:**
- [ ] Review outdated packages
- [ ] Check release notes for breaking changes
- [ ] Update core dependencies (ChromaDB, sentence-transformers, networkx)
- [ ] Test after updates: `pytest tests/ -v`
- [ ] Commit updated requirements.txt

---

### 2. Security Audit

**Why:** Detect known vulnerabilities in dependencies

```bash
# Run pip-audit (install if needed)
pip install pip-audit
pip-audit

# Check for security issues
pip-audit --desc
```

**Action Items:**
- [ ] Run pip-audit
- [ ] Review any vulnerabilities found
- [ ] Update vulnerable packages immediately
- [ ] If no fix available, document the risk
- [ ] Consider alternative packages if needed

---

### 3. Review GitHub Security Alerts

**Why:** GitHub automatically scans for vulnerabilities (Dependabot)

**Steps:**
1. Visit: https://github.com/ircaspurna/knowledge-extraction-pipeline/security
2. Check for Dependabot alerts
3. Review and merge Dependabot PRs
4. Update local repository after merge

**Action Items:**
- [ ] Check GitHub Security tab
- [ ] Review any Dependabot alerts
- [ ] Merge automated security updates
- [ ] Pull latest changes locally
- [ ] Re-run tests to verify

---

### 4. Check for Secret Exposure

**Why:** Ensure no credentials accidentally committed

```bash
# Search git history for secrets
git log --all -p | grep -i "password\|secret\|key\|api"

# Check for .env files in git
git log --all --full-history -- "*/.env"

# Use dedicated tool (optional)
pip install detect-secrets
detect-secrets scan
```

**Action Items:**
- [ ] Search git history for exposed secrets
- [ ] Verify .env files not committed
- [ ] Check recent commits for sensitive data
- [ ] Rotate any exposed credentials immediately

---

## 🧪 Code Quality Maintenance (Monthly)

### 5. Run Full Test Suite

**Why:** Catch regressions and ensure everything still works

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src/knowledge_extraction --cov-report=html

# Check coverage report
open htmlcov/index.html  # Mac
# Or: firefox htmlcov/index.html  # Linux
```

**Action Items:**
- [ ] Run full test suite
- [ ] Verify all tests pass
- [ ] Check test coverage (target: >60%)
- [ ] Fix any failing tests
- [ ] Add tests for new features

---

### 6. Run Code Quality Checks

**Why:** Maintain consistent code style and catch potential bugs

```bash
# Run linting
make lint
# Or manually:
ruff check src/ tests/ scripts/
mypy src/ tests/ scripts/

# Run formatting check
ruff format --check src/ tests/ scripts/
```

**Action Items:**
- [ ] Run ruff linting
- [ ] Run mypy type checking
- [ ] Fix any issues found
- [ ] Run formatters if needed
- [ ] Commit any formatting fixes

---

### 7. Review CI/CD Status

**Why:** Ensure automated checks are passing

**Steps:**
1. Visit: https://github.com/ircaspurna/knowledge-extraction-pipeline/actions
2. Check recent workflow runs
3. Investigate any failures

**Action Items:**
- [ ] Check GitHub Actions status
- [ ] Verify quality.yml workflow passing
- [ ] Verify type-check.yml workflow passing
- [ ] Fix any CI/CD failures
- [ ] Update workflow files if needed

---

## 📊 Performance & Monitoring (Monthly)

### 8. Review Logs for Issues

**Why:** Catch problems users might encounter

```bash
# Check for error patterns in logs
find . -name "*.log" -type f -mtime -30 -exec grep -i "error\|warning" {} +

# Review recent execution logs
# (location depends on your setup)
```

**Action Items:**
- [ ] Review recent error logs
- [ ] Identify common error patterns
- [ ] Fix recurring issues
- [ ] Update error handling if needed
- [ ] Document known issues

---

### 9. Check Disk Usage

**Why:** ChromaDB and graph files can grow large

```bash
# Check sizes
du -sh "/Users/IRI/Knowledge Base/Pipeline/Open Source/knowledge-extraction-pipeline"
du -sh chroma_db/
du -sh *.json

# Clean up if needed
make clean  # Removes cache files
```

**Action Items:**
- [ ] Check disk usage
- [ ] Clean up old cache files
- [ ] Archive old graph versions
- [ ] Compress old data if needed
- [ ] Document data retention policy

---

## 📚 Documentation Maintenance (Monthly)

### 10. Update Documentation

**Why:** Keep docs accurate and helpful

**Check:**
- README.md - Is quick start still accurate?
- CONTRIBUTING.md - Are setup instructions current?
- CHANGELOG.md - Are recent changes documented?
- Example files - Do they still work?

**Action Items:**
- [ ] Test quick start instructions
- [ ] Update version numbers
- [ ] Add new features to README
- [ ] Update CHANGELOG.md
- [ ] Fix any broken links

---

### 11. Review Open Issues

**Why:** Address user feedback and bugs

**Steps:**
1. Visit: https://github.com/ircaspurna/knowledge-extraction-pipeline/issues
2. Respond to open issues
3. Close resolved issues
4. Prioritize important bugs

**Action Items:**
- [ ] Review all open issues
- [ ] Respond to user questions
- [ ] Close resolved issues
- [ ] Label and prioritize remaining issues
- [ ] Plan fixes for critical bugs

---

## 🔄 Backup & Version Control (Monthly)

### 12. Backup Important Data

**Why:** Protect against data loss

```bash
# Backup key files
BACKUP_DIR=~/Backups/knowledge-base-$(date +%Y-%m-%d)
mkdir -p "$BACKUP_DIR"

# Copy important files
cp knowledge_graph.json "$BACKUP_DIR/"
cp -r config/ "$BACKUP_DIR/"
cp -r docs/ "$BACKUP_DIR/"

# Optional: Backup ChromaDB
tar -czf "$BACKUP_DIR/chroma_db.tar.gz" chroma_db/
```

**Action Items:**
- [ ] Backup knowledge graphs
- [ ] Backup configuration files
- [ ] Backup important documentation
- [ ] Test restoration process
- [ ] Clean up old backups (>3 months)

---

### 13. Review Git Repository

**Why:** Maintain clean version history

```bash
# Check repository health
git status
git log --oneline --graph --all -20

# Clean up if needed
git gc  # Garbage collection
git prune  # Remove unreachable objects
```

**Action Items:**
- [ ] Check for uncommitted changes
- [ ] Review recent commits
- [ ] Push any local changes
- [ ] Clean up old branches
- [ ] Tag releases if appropriate

---

## 📈 Performance Optimization (Quarterly, but check monthly)

### 14. Benchmark Performance

**Why:** Ensure pipeline isn't slowing down

```bash
# Time a typical workflow
time python scripts/process_pdf.py test.pdf --output ./test_output/

# Monitor memory usage
/usr/bin/time -l python scripts/process_pdf.py test.pdf
```

**Action Items:**
- [ ] Benchmark PDF processing time
- [ ] Check memory usage
- [ ] Compare to previous benchmarks
- [ ] Investigate any slowdowns
- [ ] Optimize if needed

---

## ✅ Monthly Checklist Summary

**Quick Reference - Copy this each month:**

```
Date: _________

Security:
[ ] Update dependencies (pip list --outdated)
[ ] Run security audit (pip-audit)
[ ] Check GitHub security alerts
[ ] Verify no secrets exposed

Code Quality:
[ ] Run full test suite (pytest -v --cov)
[ ] Run linting (make lint)
[ ] Check CI/CD status
[ ] Review and fix any issues

Monitoring:
[ ] Review logs for errors
[ ] Check disk usage
[ ] Benchmark performance

Documentation:
[ ] Update README if needed
[ ] Review and respond to issues
[ ] Update CHANGELOG

Backup:
[ ] Backup knowledge graphs
[ ] Backup configurations
[ ] Push all changes to GitHub

Notes:
_________________________________
_________________________________
_________________________________
```

---

## 🚨 Emergency Procedures

### If You Find a Security Vulnerability

1. **DO NOT** commit the fix publicly yet
2. Assess severity (can it be exploited in the wild?)
3. Prepare a fix
4. If critical:
   - Rotate any exposed credentials immediately
   - Notify users via GitHub Security Advisory
   - Release patch version ASAP
5. If minor:
   - Include in next regular release
   - Document in CHANGELOG

### If Tests Start Failing

1. Check what changed recently
2. Review recent commits
3. Test locally with verbose output
4. Rollback if necessary
5. Fix the issue
6. Add tests to prevent regression

### If Dependencies Break

1. Check dependency changelogs
2. Pin working versions temporarily
3. File issue with dependency maintainer
4. Update code to work with new version
5. Test thoroughly before releasing

---

## 📅 Recommended Schedule

### Monthly (1st of each month)
- Security updates
- Dependency updates
- Test suite
- Issue review

### Quarterly (Every 3 months)
- Major dependency updates
- Performance benchmarks
- Documentation overhaul
- Backup verification

### Yearly (Anniversary of release)
- Security audit by external party
- Architecture review
- Roadmap planning
- Major version release

---

## 🔗 Quick Links

**Repository:**
- GitHub: https://github.com/ircaspurna/knowledge-extraction-pipeline

**Security:**
- Security tab: https://github.com/ircaspurna/knowledge-extraction-pipeline/security
- Dependabot: https://github.com/ircaspurna/knowledge-extraction-pipeline/network/updates

**CI/CD:**
- GitHub Actions: https://github.com/ircaspurna/knowledge-extraction-pipeline/actions

**Documentation:**
- Main README: https://github.com/ircaspurna/knowledge-extraction-pipeline/blob/main/README.md
- Contributing Guide: https://github.com/ircaspurna/knowledge-extraction-pipeline/blob/main/CONTRIBUTING.md

---

## 📝 Maintenance Log

Keep track of when you complete maintenance:

```
2025-11: [✓] Initial setup and security hardening
2025-12: [ ]
2026-01: [ ]
2026-02: [ ]
2026-03: [ ] Quarterly review
```

---

**Remember:** Consistent small maintenance is better than irregular large overhauls!

**Time investment:** 30-45 minutes/month prevents hours of emergency fixes later.

**Contact:** For questions, open a GitHub Discussion or Issue.

---

**Last Reviewed:** 2025-11-25
**Next Review Due:** 2025-12-25
