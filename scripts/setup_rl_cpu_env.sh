#!/usr/bin/env bash
set -euo pipefail

INSTALL_DEPS="${INSTALL_DEPS:-0}"

if [[ "$INSTALL_DEPS" == "1" ]]; then
  python3 -m pip install -U numpy torch
fi

python3 - <<'PY'
import importlib
import sys

print(f"python: {sys.version.split()[0]}")
for name in ("numpy", "torch"):
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        print(f"{name}: MISSING ({exc})")
        raise SystemExit(1)
    print(f"{name}: {getattr(module, '__version__', 'ok')}")
print("RL CPU environment check passed.")
PY
