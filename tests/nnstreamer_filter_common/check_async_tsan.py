#!/usr/bin/env python3
"""@package check_async_tsan
@brief Reject an unsafe input read, then restore and verify the race gate.
"""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
build = Path(sys.argv[1] if len(sys.argv) > 1 else 'build-tsan').resolve()
source = root / 'gst/nnstreamer/tensor_filter/tensor_filter.c'
runner = root / 'tests/nnstreamer_filter_common/run-async-tsan.sh'
original = source.read_bytes()
needle = b'  if (!priv->prop.invoke_async) {\n    retval =\n        _gst_tensor_filter_transform_prepare_output_tensors'
if original.count(needle) != 1:
    raise SystemExit('Cannot locate the async input preparation guard')


def run(name, command):
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    (build / 'meson-logs' / (name + '.log')).write_text(result.stdout + result.stderr)
    print(name + ': exit ' + str(result.returncode), flush=True)
    return result


try:
    # Restore only the unsafe dimension read, retaining the exact same tests.
    source.write_bytes(original.replace(needle, needle.replace(
        b'if (!priv->prop.invoke_async)', b'if (TRUE)')))
    compiled = run('async-tsan-negative-build', ['meson', 'compile', '-C', str(build), '-j2'])
    if compiled.returncode:
        raise RuntimeError('Negative control did not compile')
    rejected = run('async-tsan-negative', ['bash', str(runner), str(build)])
    diagnostics = rejected.stdout + rejected.stderr
    if rejected.returncode == 0 or 'ThreadSanitizer: data race' not in diagnostics:
        raise RuntimeError('The gate did not report the restored NNStreamer race')
    if 'tensor_filter.c' not in diagnostics:
        raise RuntimeError('The negative control did not identify NNStreamer source')
finally:
    source.write_bytes(original)
    restored = run('async-tsan-restored-build', ['meson', 'compile', '-C', str(build), '-j2'])
    if restored.returncode:
        raise RuntimeError('Restored source did not compile')

fixed = run('async-tsan-fixed', ['bash', str(runner), str(build)])
if fixed.returncode:
    raise SystemExit(fixed.returncode)
print('Unsafe control rejected; restored source passed the same race gate.')
