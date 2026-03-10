# Publishing checklist

This snapshot is close to a GitHub launch, but the public surface is only real after you publish it.

## Launch now

### 1. Clean the tree before the first commit

Do not push local packaging leftovers such as:

- `build/`
- `dist/`
- `tsdataforge.egg-info/`
- `.pytest_cache/`
- `PKG-INFO`
- `setup.cfg`

### 2. Create the GitHub repository

If you use GitHub CLI:

```bash
git init
git branch -M main
git add .
git commit -m "Launch TSDataForge 0.3.7"
gh repo create tsdataforge --public --source=. --remote=origin --push
```

If you use the web UI:

1. create an empty public repo
2. copy the remote URL
3. run:

```bash
git init
git branch -M main
git add .
git commit -m "Launch TSDataForge 0.3.7"
git remote add origin <YOUR_GITHUB_REMOTE>
git push -u origin main
```

### 3. Turn docs and releases on

1. enable GitHub Pages from GitHub Actions
2. confirm `.github/workflows/docs.yml` publishes successfully
3. create the first GitHub release from tag `v0.3.7`

### 4. Publish to PyPI

Use your normal release workflow or trusted publish setup.
Only add public package URLs back into `pyproject.toml` after the public pages are live.

### 5. Verify the public promise

Before announcing the repo, confirm all of these are true:

- the README screenshots match the pushed version
- the default `demo` command still works from a clean checkout
- `docs/public_data_provenance.md` is visible from the README
- GitHub repo URL is live
- docs URL is live
- `pip install tsdataforge` really works

## Why this file exists

Recent product audits were right about one thing:

**broken public URLs are worse than missing ones.**

This file keeps the launch sequence explicit so the public surface only appears after it is real.
