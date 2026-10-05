import pytest


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    """Run every test in a temp dir so code that writes relative paths (e.g. artifacts/) leaves the repo clean."""
    monkeypatch.chdir(tmp_path)
