#!/usr/bin/env bash
# @file run-async-tsan.sh
# @brief Fail on a race; system GLib/GStreamer are not instrumented.
set -euo pipefail
build_dir=${1:-build-tsan}
GST_REGISTRY="$(realpath "$build_dir")/async-tsan-registry.bin"
export GST_REGISTRY
export GST_REGISTRY_FORK=no
export TSAN_OPTIONS=halt_on_error=1:exitcode=66:ignore_noninstrumented_modules=1
# GCC TSan's address map requires disabled ASLR on current Linux kernels.
exec setarch "$(uname -m)" -R meson test -C "$build_dir" --no-rebuild -v \
  unittest_filter_common \
  --test-args='--gtest_filter=testFilterAsyncOutput.concurrentLegacyMetadata --gtest_repeat=3'
