set shell := ["bash", "-uc"]

# 切回主分支并更新；若原分支与主分支内容无差异，则删除原本地分支。
cleanup-branch:
    #!/usr/bin/env bash
    set -euo pipefail

    current_branch="$(git branch --show-current)"
    if [[ -z "${current_branch}" ]]; then
      echo "当前不在命名分支上，无法自动清理。"
      exit 1
    fi

    if ! git diff --quiet || ! git diff --cached --quiet; then
      echo "存在未提交的 tracked 改动；请先提交、stash 或撤销后再运行。"
      exit 1
    fi

    main_branch=""
    for candidate in main master develop dev; do
      if git show-ref --verify --quiet "refs/heads/${candidate}"; then
        main_branch="${candidate}"
        break
      fi
    done

    if [[ -z "${main_branch}" ]]; then
      git fetch origin --prune
      for candidate in main master develop dev; do
        if git show-ref --verify --quiet "refs/remotes/origin/${candidate}"; then
          git switch --track -c "${candidate}" "origin/${candidate}"
          main_branch="${candidate}"
          break
        fi
      done
    fi

    if [[ -z "${main_branch}" ]]; then
      echo "没有找到主分支候选：main/master/develop/dev。"
      exit 1
    fi

    if [[ "${current_branch}" == "${main_branch}" ]]; then
      echo "当前已经在主分支 ${main_branch}，无需删除。"
      exit 0
    fi

    git switch "${main_branch}"
    git pull --ff-only

    if git diff --quiet "${main_branch}..${current_branch}"; then
      git branch -D "${current_branch}"
      echo "已删除本地分支 ${current_branch}；它与 ${main_branch} 内容无差异。"
    else
      echo "保留本地分支 ${current_branch}；它与 ${main_branch} 仍有内容差异。"
      echo "查看差异：git diff ${main_branch}..${current_branch}"
      exit 1
    fi
