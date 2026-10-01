# Owner-supplied snapshot check

Copyright 2026 Parth Maniar. Apache-2.0.

On October 1, 2026, Parth supplied a five-file triple-target-encoding archive for the final repository handoff. Its training source is byte-for-byte identical to `reference/scored-triple-te.py` already packaged. Its CSV is byte-for-byte identical to `predictions/submission.csv`, the 0.94614 public / 0.94518 private TE entry. There is no model-code update to merge.

The supplied README, license and both scripts are preserved unchanged in `reference/owner-supplied/`. The CSV is included once to avoid redundant large files. The portable scripts and cleaned notebooks retain the same training method with local paths.

Training source SHA256: `b43c928b2af86f37a6c2be24fe65438c7e8667ced61d5d8d5ac2cefb33a205d0`.

CSV SHA256: `339036af14e1ed28e566bd9be92dc6b192b54a7f8cd15a825c28643613abddef`.

This confirms the TE source, not the source of the final private-board winner. `predictions/final-blend.csv` remains the actual 0.94526-private counted artifact, with its exact recipe unrecovered. The supplied archive contains no blend recipe, pure-cell rules or replacement final-blend predictions. See [final-blend recovery](final-blend-recovery.md).
