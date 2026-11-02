#!/usr/bin/env python3
"""Single entry point of the P2DU pipeline.

    python run_pipeline.py                         # full run (all stages in order)
    python run_pipeline.py --stage rq1             # one stage (data | eda | rq1_fit | rq1 | rq2 | rq3 | sens_fit |
                                                   #   sensitivity | inference | figures | macros | manifest)
    python run_pipeline.py --stage rq1,rq3,figures # several stages, in this order
    python run_pipeline.py --quick                 # smoke test on a few origins (outputs are NOT the reported ones)

All parameters are in config.yaml; all outputs are written to outputs/ together with manifest.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from p2du.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
