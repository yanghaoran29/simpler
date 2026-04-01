#!/usr/bin/env bash
# Print qemu-aarch64 path that matches libinsn_count.so: same QEMU_BUILD_DIR as plugins/Makefile.
# Usage: resolve_plugin_qemu.sh <UT_DIR>
# Exit 0 always; prints one line (absolute path if possible).
set -euo pipefail
UT_DIR="${1:?UT_DIR required}"
MK="${UT_DIR}/plugins/Makefile"
if [[ -f "${MK}" ]]; then
    _qbd="$(sed -n 's/^QEMU_BUILD_DIR[[:space:]]*[?:]*=[[:space:]]*//p' "${MK}" | head -1)"
    _qbd="${_qbd%%$'\r'}"
    if [[ -n "${_qbd}" && -x "${_qbd}/qemu-aarch64" ]]; then
        readlink -f "${_qbd}/qemu-aarch64" 2>/dev/null || echo "${_qbd}/qemu-aarch64"
        exit 0
    fi
fi
# Historical default in this tree (override with QEMU_BIN or fix plugins/Makefile QEMU_BUILD_DIR).
echo "/data/y00955915/.local/bin/qemu-aarch64"
