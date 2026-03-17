# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

NNStreamer is a set of GStreamer plugins (C/C++) that integrate neural network inference into media streaming pipelines. Build system: **Meson >= 0.62.0 + Ninja**. See `Documentation/getting-started-meson-build.md` for full build instructions.

### Build & Run

```bash
# Configure (use gcc explicitly; clang may need libstdc++.so symlink fix)
CC=gcc CXX=g++ meson setup build/
meson compile -C build/

# Environment variables required to use built plugins without installing:
export GST_PLUGIN_PATH=$(pwd)/build/gst:$(pwd)/build/ext
export NNSTREAMER_CONF=$(pwd)/build/nnstreamer-test.ini
export NNSTREAMER_FILTERS=$(pwd)/build/ext/nnstreamer/tensor_filter
export NNSTREAMER_DECODERS=$(pwd)/build/ext/nnstreamer/tensor_decoder
export NNSTREAMER_CONVERTERS=$(pwd)/build/ext/nnstreamer/tensor_converter
```

### Testing

```bash
# Run all unit tests (GTest-based, 19 test suites)
meson test -C build/ -v

# Verify installation
build/tools/development/confchk/nnstreamer-check

# Quick smoke-test pipeline
gst-launch-1.0 videotestsrc num-buffers=10 ! video/x-raw,width=320,height=240,format=RGB,framerate=30/1 \
  ! tensor_converter ! tensor_sink
```

### Lint

NNStreamer uses CI-based format checking (see `.github/workflows/static.check.yml` and `.github/workflows/cpp-linter.yml`). C code follows K&R style with 2-space indentation; C++ code uses the repo `.clang-format`.

### Gotchas

- On Ubuntu 24.04, the default `cc`/`c++` may point to clang, which can fail to link with `-lstdc++` if the symlink `/usr/lib/x86_64-linux-gnu/libstdc++.so` is missing. Fix: `sudo ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/libstdc++.so`, or use `CC=gcc CXX=g++` explicitly.
- Meson is installed via pip to `~/.local/bin`. Ensure `PATH` includes `$HOME/.local/bin`.
- Most ML framework sub-plugins (TFLite, PyTorch, ONNX, etc.) are **optional**. The core build and tests work without them. The build auto-detects available frameworks via `meson_options.txt` (`auto` feature values).
- The `meson test` command sets all necessary environment variables automatically (see root `meson.build` `testenv` block). Manual pipeline testing requires the env vars listed above.
- After `meson setup`, if you change meson options, run `meson setup --reconfigure build/` instead of deleting the build directory.
