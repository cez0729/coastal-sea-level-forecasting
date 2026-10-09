# Five-minute upload

1. Create an empty GitHub repository, for example `coastal-sea-level-forecasting`.
2. Extract this folder, open PowerShell inside it, and run:

```powershell
git init
git branch -M main
git add .
git status
git commit -m "Initial reproducible coastal sea-level forecasting project"
git remote add origin https://github.com/YOUR_USER/coastal-sea-level-forecasting.git
git push -u origin main
```

3. Replace `YOUR_USER` with the GitHub account name. When GitHub asks for a password, use a Personal Access Token or SSH.

The folder is the upload root. Upload its contents, not the ZIP as a single file. `github_upload_20261009.zip` is provided only for transfer.

Raw data and checkpoints are intentionally excluded. The repository contains their source and download instructions. The selected result CSV files are under `results/tables/`.
