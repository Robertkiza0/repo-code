#!/usr/bin/env python
"""Reconstruct raw source repos referenced by the CrossCodeEval Python task set.

CrossCodeEval's line_completion.jsonl only ships prompt/groundtruth snippets,
not full repo source (the dataset README says to email the authors for that).
Each task's metadata.repository is "{owner}-{repo}-{commit7}" though, which is
just enough to reconstruct the actual repos from GitHub ourselves:

  1. Group tasks by repository id, split into (owner, repo, commit).
     The owner/repo boundary is ambiguous (both can contain hyphens), so we
     probe candidate splits with `git ls-remote` until one resolves.
  2. Partial-clone (--filter=blob:none) each resolved repo and check out the
     pinned commit. This avoids downloading full blob history -- only the
     tree/commit graph plus the blobs actually needed for that one commit.
  3. Sanity-check that at least one file path referenced by the dataset's
     metadata actually exists in the checked-out tree, to catch cases where
     ls-remote resolved to the *wrong* same-named repo.

Resumable: repos already present under the output dir (with a checked-out
.git) are skipped. Failures are logged to a report instead of aborting the
whole run, since some repos will be renamed/deleted/private by now.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9._-]+$")
LS_REMOTE_TIMEOUT = 20
CLONE_TIMEOUT = 300
CHECKOUT_TIMEOUT = 120


def load_tasks(jsonl_path: Path) -> Dict[str, Dict]:
    """repository_id -> {"commit": str, "files": set[str]}"""
    repos: Dict[str, Dict] = {}
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            meta = task["metadata"]
            repo_id = meta["repository"]
            entry = repos.setdefault(repo_id, {"commit": None, "files": set()})
            entry["files"].add(meta["file"])
    return repos


def split_repo_id(repo_id: str) -> Tuple[List[str], str]:
    tokens = repo_id.split("-")
    if len(tokens) < 3 or not COMMIT_RE.match(tokens[-1]):
        raise ValueError(f"cannot find a commit suffix in repository id {repo_id!r}")
    return tokens[:-1], tokens[-1]


def candidate_owner_repo_splits(name_tokens: List[str]) -> List[Tuple[str, str]]:
    candidates = []
    for owner_len in range(1, len(name_tokens)):
        owner = "-".join(name_tokens[:owner_len])
        repo = "-".join(name_tokens[owner_len:])
        candidates.append((owner, repo))
    return candidates


def remote_exists(owner: str, repo: str) -> bool:
    if not (SAFE_TOKEN_RE.match(owner) and SAFE_TOKEN_RE.match(repo)):
        return False
    url = f"https://github.com/{owner}/{repo}.git"
    try:
        result = subprocess.run(
            ["git", "ls-remote", "--exit-code", url, "HEAD"],
            capture_output=True,
            timeout=LS_REMOTE_TIMEOUT,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def resolve_owner_repo(repo_id: str) -> Tuple[str, str, str]:
    name_tokens, commit = split_repo_id(repo_id)
    for owner, repo in candidate_owner_repo_splits(name_tokens):
        if remote_exists(owner, repo):
            return owner, repo, commit
    raise LookupError(f"no owner/repo split of {repo_id!r} resolves on GitHub")


def clone_and_checkout(owner: str, repo: str, commit: str, dest: Path) -> None:
    url = f"https://github.com/{owner}/{repo}.git"
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "-q", url, str(dest)],
        check=True,
        capture_output=True,
        timeout=CLONE_TIMEOUT,
    )
    subprocess.run(
        ["git", "checkout", "-q", commit],
        check=True,
        capture_output=True,
        timeout=CHECKOUT_TIMEOUT,
        cwd=str(dest),
    )


def sanity_check(dest: Path, files: Set[str]) -> bool:
    return any((dest / f).is_file() for f in files)


def already_fetched(dest: Path) -> bool:
    return (dest / ".git").exists()


def fetch_all(jsonl_path: Path, output_dir: Path, limit: Optional[int] = None) -> Dict:
    repos = load_tasks(jsonl_path)
    repo_ids = sorted(repos)
    if limit:
        repo_ids = repo_ids[:limit]

    resolved: List[str] = []
    skipped: List[str] = []
    failed: Dict[str, str] = {}
    suspicious: List[str] = []

    total = len(repo_ids)
    for i, repo_id in enumerate(repo_ids, 1):
        dest = output_dir / repo_id
        prefix = f"[{i}/{total}] {repo_id}"

        if already_fetched(dest):
            print(f"{prefix} -> already fetched, skipping")
            skipped.append(repo_id)
            continue

        try:
            owner, repo, commit = resolve_owner_repo(repo_id)
            clone_and_checkout(owner, repo, commit, dest)
            if sanity_check(dest, repos[repo_id]["files"]):
                print(f"{prefix} -> ok ({owner}/{repo}@{commit})")
                resolved.append(repo_id)
            else:
                print(f"{prefix} -> cloned but no expected files found (possible wrong repo)")
                suspicious.append(repo_id)
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="replace").strip() if e.stderr else str(e)
            print(f"{prefix} -> FAILED: {stderr.splitlines()[-1] if stderr else e}")
            failed[repo_id] = stderr
        except Exception as e:  # noqa: BLE001 - keep the batch going on any single-repo failure
            print(f"{prefix} -> FAILED: {e}")
            failed[repo_id] = str(e)

    report = {
        "total": total,
        "resolved": len(resolved),
        "skipped_already_present": len(skipped),
        "suspicious": suspicious,
        "failed_count": len(failed),
        "failed": failed,
    }
    report_path = output_dir / "_fetch_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nDone. resolved={len(resolved)} skipped={len(skipped)} suspicious={len(suspicious)} failed={len(failed)}")
    print(f"Report: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", default="data/python/line_completion.jsonl")
    parser.add_argument("--output", default="data/raw/python")
    parser.add_argument("--limit", type=int, default=None, help="Only fetch the first N repos (for testing)")
    args = parser.parse_args()

    fetch_all(Path(args.jsonl), Path(args.output), args.limit)


if __name__ == "__main__":
    main()
