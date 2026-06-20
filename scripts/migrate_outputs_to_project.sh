#!/usr/bin/env bash
set -euo pipefail

PUBLIC_ROOT="${PUBLIC_ROOT:-/inspire/hdd/project/generative-large-model/public}"
PROJECT_ROOT="${PROJECT_ROOT:-${PUBLIC_ROOT}/hw3-of-lpf}"
OLD_OUTPUTS="${OLD_OUTPUTS:-${PUBLIC_ROOT}/outputs}"
NEW_OUTPUTS="${NEW_OUTPUTS:-${PROJECT_ROOT}/outputs}"

mkdir -p "$NEW_OUTPUTS"

if [[ ! -d "$OLD_OUTPUTS" ]]; then
  echo "No old output directory found: $OLD_OUTPUTS"
  echo "Project output directory is ready: $NEW_OUTPUTS"
  exit 0
fi

echo "Copying existing results into the project directory."
echo "  from: $OLD_OUTPUTS"
echo "  to:   $NEW_OUTPUTS"
echo

shopt -s nullglob
for src in "$OLD_OUTPUTS"/*; do
  name="$(basename "$src")"
  dst="$NEW_OUTPUTS/$name"
  if [[ -e "$dst" ]]; then
    echo "skip existing: $dst"
    continue
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "would copy: $src -> $dst"
  else
    echo "copy: $src -> $dst"
    cp -a "$src" "$dst"
  fi
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  chmod -R a+rwX "$NEW_OUTPUTS"
fi

echo
echo "Done. New results should live under:"
echo "  $NEW_OUTPUTS"
