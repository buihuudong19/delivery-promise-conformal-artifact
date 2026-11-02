"""Clear the results of the previous run before a full run, so that every number of the new run is fresh.

Called by run_pipeline.bat and run_pipeline.sh before a full run (no arguments). Uses the standard library only.

The contents of outputs/ (tables, figures, logs, fitted models in outputs/data/parts, manuscript files, manifest) and of
outputs_extra/ (independent cross-checks) are moved to _archive/run_<date>_<time>/outputs and .../outputs_extra, so the
new run starts from empty folders and refits every model. Only the KEEP most recent archives are kept; older ones are
deleted, so the archive does not grow without bound (python tools/clear_previous_run.py --keep 5 keeps five).

If an item cannot be moved (on Windows usually because a file is open in another program, or a python.exe of the
previous run is still alive), the items moved so far are put back, the folders are left as they were, and the script
exits with status 1 so that the run does not start on a mixed folder.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("outputs", "outputs_extra")
KEEP = 2


def _prune(arch, keep):
    """Delete all but the `keep` most recent run archives (names sort by date and time)."""
    runs = sorted(p for p in arch.glob("run_*") if p.is_dir()) + sorted(p for p in arch.glob("outputs_*") if p.is_dir())
    runs = sorted(runs, key=lambda p: p.name.split("_", 1)[1])
    for p in runs[:-keep] if keep > 0 else runs:
        try:
            shutil.rmtree(p)
            print(f"Deleted the old archive {p.name}")
        except OSError as e:
            print(f"Could not delete the old archive {p.name} ({e}); it is left in place.", file=sys.stderr)


def archive(root=ROOT, keep=KEEP):
    items = [(f, p) for f in FOLDERS if (root / f).is_dir() for p in sorted((root / f).iterdir())]
    if not items:
        print("outputs/ and outputs_extra/ are empty; nothing to clear.")
        return 0
    dest = root / "_archive" / f"run_{time.strftime('%Y-%m-%d_%H%M%S')}"
    dest.mkdir(parents=True, exist_ok=False)
    moved = []
    try:
        for f, p in items:
            (dest / f).mkdir(exist_ok=True)
            shutil.move(str(p), str(dest / f / p.name))
            moved.append((f, p.name))
    except OSError as e:
        for f, name in reversed(moved):                    # roll back, so that the folders are unchanged
            shutil.move(str(dest / f / name), str(root / f / name))
        shutil.rmtree(dest, ignore_errors=True)
        print(f"Could not move {p.relative_to(root)} ({e}).\n"
              "Close any program that has a file of outputs/ open (PDF viewer, Excel, Explorer preview), end any\n"
              "python.exe left from the previous run (Task Manager), and run again.", file=sys.stderr)
        return 1
    print(f"Previous results cleared: outputs/ and outputs_extra/ moved to {dest.relative_to(root)}")
    _prune(root / "_archive", keep)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--keep", type=int, default=KEEP, help="number of run archives to keep (default 2)")
    sys.exit(archive(keep=ap.parse_args().keep))
