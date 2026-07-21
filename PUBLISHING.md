# Publishing checklist

Everything below is prepared in the repo; these are the account-gated steps that
only you can run (they need your GitHub / Hugging Face credentials). Run them
from the repo root.

> Your local `gh` keyring auth is currently broken — re-authenticate first:
> `gh auth login` (or use the `git` commands directly with a Personal Access Token).

## 1. Create the GitHub repo and push (main + tag only)

The commit history author is `mathimanichandan@gmail.com`. A safety branch
`backup-before-email-rewrite` holds the pre-rewrite history — **do not push it.**

```bash
# create the public repo and push main + the release tag
gh repo create mathi0405/drone-swarm-sar --public --source=. --remote=origin --push
git push origin v0.1.0

# (manual alternative, if not using gh)
git remote add origin https://github.com/mathi0405/drone-swarm-sar.git
git push -u origin main
git push origin v0.1.0
```

CI (`.github/workflows/ci.yml`) runs on push: ruff + mypy(advisory) + pytest +
smoke demo. Watch it go green under the Actions tab.

## 2. Enable GitHub Pages for the docs site

Repo → Settings → Pages → Build and deployment → Source: **GitHub Actions**.
The `Docs` workflow then publishes to `https://mathi0405.github.io/drone-swarm-sar`
on every push to `docs/**` (or run it manually via *Actions → Docs → Run workflow*).

## 3. Cut the v0.1.0 release with the model zoo

The 10 trained checkpoints are git-ignored (they're large); attach them to the
release so the model-zoo download links resolve.

```bash
# collect the best checkpoints with zoo-friendly names
mkdir -p release_assets
for d in results/trained/*/; do
  name=$(basename "$d")
  cp "$d/checkpoints/best.pt" "release_assets/${name}_best.pt" 2>/dev/null || true
done

gh release create v0.1.0 release_assets/*_best.pt \
  --title "Swarm-SAR v0.1.0" \
  --notes-file CHANGELOG.md
```

Then regenerate the model-zoo download links against the real run names:
`python scripts/generate_model_cards.py --release-tag v0.1.0` and commit the diff.

## 4. Deploy the live demo to Hugging Face Spaces

```bash
huggingface-cli login
huggingface-cli repo create swarm-sar --type space --space_sdk streamlit
git clone https://huggingface.co/spaces/<your-hf-username>/swarm-sar hf-space
cp spaces/app.py spaces/requirements.txt spaces/README.md hf-space/
cd hf-space && git add . && git commit -m "Swarm-SAR demo" && git push
```

The Space builds on the free CPU tier (self-contained sim only) and gives you a
clickable URL to put in the README badge and the paper.

## 5. (Optional) Purge the old email from history permanently

The old-email commits survive only on the local `backup-before-email-rewrite`
branch and in `refs/original`. They never leave your machine unless pushed. Once
you've confirmed the pushed history looks right:

```bash
git branch -D backup-before-email-rewrite
git for-each-ref --format='%(refname)' refs/original | xargs -n1 git update-ref -d
git reflog expire --expire=now --all && git gc --prune=now
```

## Done — then add these to the README

- CI status badge: `![CI](https://github.com/mathi0405/drone-swarm-sar/actions/workflows/ci.yml/badge.svg)`
- Live demo badge linking to the HF Space
- Docs link: https://mathi0405.github.io/drone-swarm-sar
