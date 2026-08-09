# ADR-0001: Installable package with a Ports & Adapters core

## Status

> **Hinweis:** Paket und Kommando hiessen damals `yoloe_lane` bzw.
> `yoloe-lane`. Umbenannt in `adascope` mit [ADR-0003](ADR-0003-rename-and-flat-layout.md).

Accepted (2026-06-06)

## Context

The original layout shipped the domain logic as module-level functions inside
`scripts/detect.py` and relied on `scripts/_path.py` to inject `src/` onto
`sys.path` so scripts could `from core import load_model`. The project was never
installed (`pip install -e .`), and the `src/core` package the shim pointed at was
never created — so the pipeline crashed at step 7 with
`ModuleNotFoundError: No module named 'core'`.

Further problems: config loading, frame/video I/O, and the supported image
extensions were each re-implemented across 2–4 scripts; data was passed around as
untyped dicts; there were no tests; and `detect.py --frames` silently ignored its
`--out-video` flag.

## Decision

Restructure into a modular monolith with a **Ports & Adapters (hexagonal)** core,
shipped as the installable package `yoloe_lane` (src-layout).

- **Pure domain core** — `config`, `geometry`, `carpet`, `analysis`, `render` —
  depends only on numpy/OpenCV, never on `ultralytics` or argparse.
- **Detector port** (`detector.Detector` Protocol) decouples analysis from the
  model; `YoloeDetector` is the adapter and is the single owner of the YOLOE
  text-prompt setup (`get_text_pe` + `set_classes`). This replaces the cosmetic
  `core.load_model` wrapper-package.
- **One I/O module** (`frames`) owns extension filtering, `imread` guards, and
  video read/write.
- **Typed contracts**: `Config`/`ModelCfg`/`CarpetCfg` and `FrameResult`
  dataclasses replace ad-hoc dicts; the config is validated once at load.
- **Thin CLI adapters** under `yoloe_lane.cli`, dispatched by a single
  `python -m yoloe_lane <command>` / `yoloe-lane <command>` entry point.
- `pip install -e .` makes `import yoloe_lane` work everywhere — **`_path.py` is
  deleted**.

## Consequences

- The domain logic is unit-testable with a `FakeDetector` (no 70 MB model load);
  a `tests/` suite locks geometry, carpet, config, analysis, and frame I/O.
- New behaviour is added in one obvious place instead of being copy-pasted across
  scripts.
- Many files moved at once; behaviour parity is guaranteed by the test suite plus
  single-frame and CSV diffs against the pre-refactor run (identical output).
- `python scripts/X.py` invocations are replaced by `yoloe-lane X`;
  `run_pipeline.py` remains as a thin backwards-compatible shim.
- GUI tools (`roi-editor`, `crop-box`) can't be auto-tested beyond their headless
  `--dump` path; they share the config/geometry modules but still need a manual
  smoke test.
