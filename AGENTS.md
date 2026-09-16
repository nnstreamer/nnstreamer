# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

NNStreamer is a set of GStreamer plugins (C/C++) that integrate neural network inference into media streaming pipelines. Build system: **Meson >= 0.57.0 + Ninja**. See `Documentation/getting-started-meson-build.md` for full build instructions.

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

## Code Review Rules

### Scope and correctness

- Analyze the PR diff, relevant repository context, and relevant external-library behavior, but report findings only when the defect is introduced or exposed by the PR's changed code.
- Verify that the change solves the PR's stated problem and does not regress existing behavior, performance, compatibility, or other modules. Flag changes that are disproportionate to the stated topic or touch unrelated modules without a demonstrated need.

### Tests and CI

- Require tests that validate the changed behavior and protect against foreseeable regressions caused by later work in interacting modules. Treat missing test coverage as blocking when either code coverage or functional coverage for the changed behavior is below 80%.
- Treat as blocking when a test can detect a defect but does not make CI or the build fail, allowing the defective change to merge.
- Do not flag multiple tests exercising the same feature unless they are exact duplicates. Do not use review findings for formatting, lint, or other deterministic CI checks.

### Architecture, documentation, and security

- For architecture or API changes, require the corresponding documentation in the PR or evidence that the documentation is already updated and remains accurate.
- Flag security vulnerabilities and potential backdoors. For a credible backdoor concern, explicitly include `@myungjoo-ham` in the PR comment.

### Review reporting and commit structure

- Every finding must state severity and why it matters, and must clearly say that the comment is an AI-agent review finding. For a non-blocking (NIT) finding, include a single short sentence assessing impact.
- Treat an excessive number of commits relative to the change's complexity, unrelated topics combined in one commit, or a fix for a particular commit split into a later corrective commit as a review concern. Separate feature and test commits are acceptable, and a single topic may span multiple coherent commits.
