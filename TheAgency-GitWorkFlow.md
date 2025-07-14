****# TheAgency Git Workflow Guide

This document outlines the Git workflow used by our team. Please follow these conventions to ensure smooth collaboration across repositories and submodules.

---

## 🔧 1. Repository Structure

- The **main repository** is `TheAgency`.
- All submodules must be hosted under the **organization** `sparx-mth` and **not** under personal GitHub accounts.
  - ✅ This ensures all team members have access rights.
  - Example submodule: `https://github.com/sparx-mth/TheAgency/some_submodule.git`

---

## 🌿 2. Branching Strategy

Each developer works on their own **feature branch** inside the relevant repository (main or submodule).

### 🧠 Naming Convention for Feature Branches:

```text
<short_feature_name>_<your_name>
```


Examples:
- `robot_coloring_shir`
- `autonomous_nav_nadav`
- `map_merger_daphna`

---

## 🚀 3. Feature Completion & Merging

When you reach a **significant milestone** (feature completed or major step done):

1. Merge your feature branch into the **main branch** of the repo (`main` or `master`, depending on the repo).
2. If it's a submodule:
   - Push your changes
   - Update the submodule pointer in the main repository (`TheAgency`) to the new commit.
3. Optionally, delete your feature branch to avoid clutter.

---

## 🧭 4. Workflow Steps (with Git commands)

Here’s a complete step-by-step for handling updates:

### ✅ A. Commit your local work to your feature branch:

```bash
git add .
git commit -m "Finished feature X"
```
### 🔄 B. Sync your feature branch with the latest main:
``` bash
git checkout <your-branch>
git fetch origin
git merge origin/main
```
Resolve any conflicts locally.

### 🔀 C. Merge your feature branch into main:
``` bash
git checkout main
git pull origin main
git merge <your-branch>
git commit -m "Merge feature: <your-branch>"
git push origin main
```

### 📦 D. If working on a submodule
1. Push your changes in the submodule:
```bash
git push origin main
```
2. Go to the main repo (TheAgency) and update the submodule pointer
```bash
cd path/to/submodule
git checkout main
git pull

cd ../..  # back to TheAgency
git add path/to/submodule
git commit -m "Update submodule <name> to latest commit"
git push origin main

```

### 🧹 E. Clean up your finished branch:
```bash
git branch -d <your_branch>
git push origin --delete <your_branch>
```


### 💡 Notes and Tips
* Always pull from main before merging to avoid conflicts.

* Push only when your local main has the most updated code.

* If multiple people are working on the same submodule, coordinate before pushing changes.

* Don’t forget to update the submodule pointer in TheAgency after a submodule commit.


### 🛠️ Submodule Initialization Tip
To clone TheAgency and initialize all submodules:

```bash 
git clone --recurse-submodules git@github.com:sparx-mth/TheAgency.git

```
If you already cloned without submodules:
```bash 
git submodule update --init --recursive
```

## 🧠 Suggested Improvements (Optional)
1. **Tag important releases**  
   You can optionally tag stable commits:
   ```bash
   git tag -a v1.0 -m "Initial release"
   git push origin v1.0
   ```
2. **Use Pull Requests (PRs)** for merging into `main`  
   This helps track discussions, enables code review, and triggers CI checks (if used).****