# Parameters: Optional commit message
param (
    [string]$CommitMessage = "Auto-commit: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
)

# 1. Check if the current directory is a Git repository
if (-not (Test-Path ".git")) {
    Write-Error "Error: Current directory is not a Git repository."
    exit 1
}

# 2. Check for changes (Modified, Deleted, or Untracked files)
$status = git status --porcelain
if (-not $status) {
    Write-Host "No changes detected. Nothing to commit." -ForegroundColor Cyan
    exit 0
}

Write-Host "Changes detected. Starting auto-sync..." -ForegroundColor Yellow

# 3. Stage all changes
Write-Host "Staging changes..."
git add -A

# 4. Commit changes
Write-Host "Committing with message: '$CommitMessage'"
git commit -m $CommitMessage

# 5. Get current branch name and push
$branch = git branch --show-current
Write-Host "Pushing to remote branch: $branch..."
git push origin $branch

if ($LASTEXITCODE -eq 0) {
    Write-Host "Successfully pushed changes!" -ForegroundColor Green
} else {
    Write-Error "Failed to push changes to remote."
}