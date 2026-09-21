#!/usr/bin/env bash
set -euo pipefail

OWNER="mauricio-herrera"
REPO="instrumental-spillover-networks"
FULL_REPO="$OWNER/$REPO"
TAG="v1.0.0"
DESCRIPTION="Reproducibility package for instrumental identification and weak-instrument-robust inference in high-frequency spillover networks with latent common drivers."

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v gh >/dev/null || { echo "GitHub CLI (gh) is required" >&2; exit 1; }

gh auth status

if [[ ! -d .git ]]; then
  git init -b main
fi

git add .
if ! git diff --cached --quiet; then
  git commit -m "Release v1.0.0: frozen reproducibility package"
fi

if gh repo view "$FULL_REPO" >/dev/null 2>&1; then
  echo "Repository already exists: https://github.com/$FULL_REPO"
  if ! git remote get-url origin >/dev/null 2>&1; then
    git remote add origin "https://github.com/$FULL_REPO.git"
  fi
  git push -u origin main
else
  gh repo create "$FULL_REPO" \
    --public \
    --description "$DESCRIPTION" \
    --source . \
    --remote origin \
    --push
fi

# Create or reuse annotated tag.
if ! git rev-parse "$TAG" >/dev/null 2>&1; then
  git tag -a "$TAG" -m "Frozen reproducibility release $TAG"
  git push origin "$TAG"
fi

# Create a clean archival ZIP from the tagged repository state.
ARCHIVE="../${REPO}-${TAG}.zip"
git archive --format=zip --output "$ARCHIVE" "$TAG"

if gh release view "$TAG" --repo "$FULL_REPO" >/dev/null 2>&1; then
  echo "Release $TAG already exists."
else
  gh release create "$TAG" \
    "$ARCHIVE" \
    --repo "$FULL_REPO" \
    --title "$TAG — Frozen reproducibility release" \
    --notes-file RELEASE_NOTES_v1.0.0.md
fi

echo
echo "GitHub publication complete:"
echo "  https://github.com/$FULL_REPO"
echo "  https://github.com/$FULL_REPO/releases/tag/$TAG"
echo "Archive: $ARCHIVE"
