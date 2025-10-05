import tempfile
from pathlib import Path


def test_imports_and_run_smoke():
    # Ensure modules import and the smoke runner executes
    from run_smoke import run

    # run will create its own temp dirs
    run()
