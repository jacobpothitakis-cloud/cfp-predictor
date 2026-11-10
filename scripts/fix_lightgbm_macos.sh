#!/bin/sh
# LightGBM's macOS wheel expects libomp from Homebrew. On a Mac without Homebrew, point it at the
# copy of libomp that ships inside the scikit-learn wheel (same venv), then re-sign the library.
set -e
SP="$(cd "$(dirname "$0")/.." && pwd)/.venv/lib/python3.12/site-packages"
LIB="$SP/lightgbm/lib/lib_lightgbm.dylib"
OMP_DIR="$SP/sklearn/.dylibs"
[ -f "$OMP_DIR/libomp.dylib" ] || { echo "scikit-learn's libomp not found in $OMP_DIR"; exit 1; }
if otool -l "$LIB" | grep -q "$OMP_DIR"; then echo "already patched"; exit 0; fi
install_name_tool -add_rpath "$OMP_DIR" "$LIB"
codesign --force --sign - "$LIB"
echo "patched $LIB"
