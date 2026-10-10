"""The ledger's source-provenance check (lemma #8) reads the git state of
ledger.REPO_ROOT. Tests must not depend on the developer's working tree, so
every test sees a clean throwaway checkout unless it builds its own."""

import subprocess

import pytest

from lemma import ledger


def git_repo(path, *, commit=True):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "t"], check=True)
    (path / "src").mkdir(exist_ok=True)
    (path / "src" / "evaluator.py").write_text("print('v1')\n")
    if commit:
        subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "v1"], check=True)
    return path


@pytest.fixture(autouse=True)
def clean_repo(tmp_path, monkeypatch):
    repo = git_repo(tmp_path / "repo")
    monkeypatch.setattr(ledger, "REPO_ROOT", repo)
    return repo
