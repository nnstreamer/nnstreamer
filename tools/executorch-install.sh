#!/usr/bin/env bash
##
# @file     executorch-install.sh
# @brief    Build and install the ExecuTorch C++ runtime so that
#           -Dexecutorch-support=enabled has something to find.
# @see      https://github.com/nnstreamer/nnstreamer/issues/4289
#
# usage: tools/executorch-install.sh [install-prefix] [git-tag]
#
# ExecuTorch ships no prebuilt desktop binaries and its pip wheel carries
# headers only, so the runtime must be built from source. Upstream generates
# executorch.pc for us, but only under EXECUTORCH_BUILD_SHARED, and that file
# needs one fixup; see the comment above the sed below.

set -e -o pipefail

PREFIX=${1:-/usr/local}
ET_TAG=${2:-v1.4.1}
WORK_DIR=${EXECUTORCH_WORK_DIR:-$(mktemp -d)}

# ExecuTorch refuses to configure unless its source directory is named exactly
# 'executorch' (pytorch/executorch#6475).
SRC_DIR="${WORK_DIR}/executorch"

if ! command -v cmake > /dev/null; then
  echo "::error::cmake is required to build ExecuTorch."
  exit 1
fi

# ExecuTorch declares cmake_minimum_required(3.24); Ubuntu 22.04 ships 3.22 and
# 24.04 ships 3.28, so a distro cmake may or may not do. Fail with the reason
# rather than with a wall of cmake policy errors.
CMAKE_VER=$(cmake --version | head -1 | awk '{print $3}')
if [ "$(printf '%s\n3.24.0\n' "${CMAKE_VER}" | sort -V | head -1)" != "3.24.0" ]; then
  echo "::error::ExecuTorch needs cmake 3.24 or newer, found ${CMAKE_VER}. Try 'pip install cmake'."
  exit 1
fi

if [ ! -d "${SRC_DIR}" ]; then
  git clone --depth 1 -b "${ET_TAG}" https://github.com/pytorch/executorch.git "${SRC_DIR}"
  git -C "${SRC_DIR}" submodule update --init --recursive --depth 1
fi

# EXECUTORCH_BUILD_SHARED is what produces libexecutorch.so,
# libexecutorch_portable_ops.so and executorch.pc. EXTENSION_MODULE is the API
# the subplugin is written against, and it in turn requires the data loader,
# the flat tensor reader and the named data map.
cmake -B "${SRC_DIR}/cmake-out" -S "${SRC_DIR}" \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DEXECUTORCH_BUILD_SHARED=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_PORTABLE_OPS=ON \
  -DEXECUTORCH_BUILD_PYBIND=OFF \
  -DBUILD_TESTING=OFF

cmake --build "${SRC_DIR}/cmake-out" -j"$(nproc 2> /dev/null || sysctl -n hw.ncpu)"
cmake --install "${SRC_DIR}/cmake-out"

PC_FILE="${PREFIX}/lib/pkgconfig/executorch.pc"
if [ ! -f "${PC_FILE}" ]; then
  echo "::error::${PC_FILE} was not installed; EXECUTORCH_BUILD_SHARED did not take effect."
  exit 1
fi

# The generated .pc links libexecutorch.so alone, which holds the runtime and
# the extensions but none of the operator kernels. A subplugin built against it
# links cleanly and then fails on the first invoke with an unregistered
# operator. The kernels live in libexecutorch_portable_ops.so and are reached
# only through static registrars, so --no-as-needed is required to keep the
# linker from dropping a library whose symbols are never referenced.
case "$(uname -s)" in
  Darwin)
    # The Mach-O linker has no --as-needed, and it keeps every library named on
    # the command line.
    sed -i '' 's|^Libs:.*|Libs: -L${libdir} -lexecutorch_portable_ops -lexecutorch|' "${PC_FILE}"
    ;;
  *)
    sed -i 's|^Libs:.*|Libs: -L${libdir} -Wl,--no-as-needed -lexecutorch_portable_ops -Wl,--as-needed -lexecutorch|' "${PC_FILE}"
    ;;
esac

echo "ExecuTorch ${ET_TAG} installed under ${PREFIX}"
