#!/bin/bash

# Remove the old backup refs if they exist
rm -rf "$(git rev-parse --git-dir)/refs/original/"

# Filter the Git history
git filter-branch -f --env-filter '
CORRECT_NAME="yuxuehui"
CORRECT_EMAIL="1170574199@qq.com"

# Always set the author and committer to the correct values
export GIT_AUTHOR_NAME="$CORRECT_NAME"
export GIT_AUTHOR_EMAIL="$CORRECT_EMAIL"
export GIT_COMMITTER_NAME="$CORRECT_NAME"
export GIT_COMMITTER_EMAIL="$CORRECT_EMAIL"
' --tag-name-filter cat -- --all

# Remove the backup refs
git for-each-ref --format="%(refname)" refs/original/ | xargs -n 1 git update-ref -d

# Clean up and consolidate the repository
git gc --prune=now --aggressive

# Force push all branches
git push --force --all