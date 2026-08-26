#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file  tizen-build-test.sh
## @date  26 Aug 2026
## @brief Build-test Tizen native apps with the Tizen Studio CLI installed
##        in this docker image, without requiring signing certificates.
##
##        Base check: a template project whose source is replaced with a
##        minimal nnstreamer C-API consumer is built, verifying the
##        toolchain, the platform rootstrap, and the ML API that
##        nnstreamer-based Tizen apps depend on.
##
##        NNS_RPM_DIR (optional): a directory with nnstreamer devel RPMs
##        built from this repository (e.g., a GBS output tree). Their
##        headers are overlaid onto the platform rootstrap and additional
##        C and C++ translation units compile against them, so header
##        changes in this repository that break Tizen apps are caught here.
##
##        NNS_EXAMPLE_DIR (optional): a Tizen native app project directory
##        (e.g., from nnstreamer-example). The project is copied, retargeted
##        to TIZEN_PROFILE, and built against the (possibly overlaid)
##        rootstrap.
## @see   https://github.com/nnstreamer/nnstreamer
## @author MyungJoo Ham <myungjoo.ham@samsung.com>

set -euo pipefail

TIZEN_PROFILE="${TIZEN_PROFILE:-tizen-10.0}"
TIZEN_ARCH="${TIZEN_ARCH:-arm}"
TIZEN_COMPILER="${TIZEN_COMPILER:-gcc}"
TIZEN_TEMPLATE="${TIZEN_TEMPLATE:-ServiceApp}"
NNS_RPM_DIR="${NNS_RPM_DIR:-}"
NNS_EXAMPLE_DIR="${NNS_EXAMPLE_DIR:-}"

case "${TIZEN_ARCH}" in
  arm) rootstrap_name="${TIZEN_PROFILE}-device.core"; rpm_arch="armv7l" ;;
  aarch64) rootstrap_name="${TIZEN_PROFILE}-device64.core"; rpm_arch="aarch64" ;;
  x86_64) rootstrap_name="${TIZEN_PROFILE}-emulator64.core"; rpm_arch="x86_64" ;;
  *) echo "Unsupported TIZEN_ARCH: ${TIZEN_ARCH}"; exit 1 ;;
esac
rootstrap="${HOME}/tizen-studio/platforms/${TIZEN_PROFILE}/tizen/rootstraps/${rootstrap_name}"
test -d "${rootstrap}"

workdir="$(mktemp -d)"
trap 'rm -rf "${workdir}"' EXIT

if [ -n "${NNS_RPM_DIR}" ]; then
  echo "Overlaying nnstreamer devel RPMs from ${NNS_RPM_DIR} onto ${rootstrap_name}"
  found=0
  for r in "${NNS_RPM_DIR}"/nnstreamer*devel*."${rpm_arch}".rpm; do
    test -f "$r" || continue
    found=1
    echo " * $r"
    (cd "${rootstrap}" && rpm2cpio "$r" | cpio -idmu --quiet)
  done
  test "${found}" = "1"
fi

tizen version
tizen create native-project -p "${TIZEN_PROFILE}" -t "${TIZEN_TEMPLATE}" \
    -n buildtest -- "${workdir}"
cd "${workdir}/buildtest"

# The rootstrap ships the ML API headers in usr/include/nnstreamer, which
# the platform default include list already covers; the un-prefixed form
# below is the one Documentation/nnstreamer_capi.md prescribes for apps.
cat > src/buildtest.c <<'EOF'
/**
 * @file  buildtest.c
 * @brief Minimal nnstreamer C-API consumer for the docker build test.
 */
#include <nnstreamer.h>
#include <nnstreamer-single.h>

int
main (void)
{
  ml_pipeline_h pipe = NULL;
  int status = ml_pipeline_construct ("videotestsrc ! fakesink", NULL, NULL,
      &pipe);
  if (status == ML_ERROR_NONE)
    ml_pipeline_destroy (pipe);
  return 0;
}
EOF

if [ -n "${NNS_RPM_DIR}" ]; then
  # One translation unit per overlaid devel header: a header is only
  # proven self-contained when it is included in isolation. Sharing a
  # translation unit lets one header's includes satisfy another's missing
  # ones, which would hide exactly the defect this check looks for. glib
  # is the documented prerequisite of these headers, hence its include
  # directories below.
  # Every devel header of packaging/nnstreamer.spec that a Tizen native
  # app can include. The plugin-development headers pulling in <gst/gst.h>
  # (tensor_if.h, tensor_converter_custom.h, tensor_decoder_custom.h,
  # nnstreamer_plugin_api.h and its decoder/converter variants) are left
  # out: the Tizen native rootstrap ships no GStreamer development
  # headers, and the GBS build above already compiles them.
  devel_headers="tensor_filter_custom.h tensor_filter_custom_easy.h \
      nnstreamer_plugin_api_trainer.h nnstreamer_util.h \
      nnstreamer_internal.h tensor_filter_single.h tensor_typedef.h \
      nnstreamer_plugin_api_filter.h nnstreamer_plugin_api_util.h \
      nnstreamer_version.h tensor_filter_cpp.hh \
      nnstreamer_cppplugin_api_filter.hh"
  devel_srcs=""
  index=0
  for header in ${devel_headers}; do
    index=$((index + 1))
    case "${header}" in
      *.hh) src="src/hdr_${index}.cc" ;;
      *) src="src/hdr_${index}.c" ;;
    esac
    cat > "${src}" <<EOF
/**
 * @file  $(basename "${src}")
 * @brief Self-containedness check for ${header}.
 */
#include <glib.h>
#include <nnstreamer/${header}>

int
nns_header_check_${index} (void)
{
  return 0;
}
EOF
    devel_srcs="${devel_srcs} ${src}"
  done

  grep -q '^USER_SRCS =' project_def.prop
  sed -i "s|^USER_SRCS =\(.*\)|USER_SRCS =\1${devel_srcs}|" project_def.prop
  grep -q '^USER_INC_DIRS =' project_def.prop
  sed -i "s|^USER_INC_DIRS =\(.*\)|USER_INC_DIRS =\1 ${rootstrap}/usr/include/glib-2.0 ${rootstrap}/usr/lib/glib-2.0/include|" project_def.prop
fi

grep -q '^USER_LIBS =' project_def.prop
sed -i 's/^USER_LIBS =\(.*\)/USER_LIBS =\1 capi-nnstreamer/' project_def.prop

tizen build-native -a "${TIZEN_ARCH}" -c "${TIZEN_COMPILER}" -C Debug
test -f Debug/buildtest
echo "Tizen native app build test: PASS"

if [ -n "${NNS_EXAMPLE_DIR}" ]; then
  app_name="$(basename "${NNS_EXAMPLE_DIR}")"
  echo "Building example app: ${app_name}"
  cp -r "${NNS_EXAMPLE_DIR}" "${workdir}/example-app"
  cd "${workdir}/example-app"
  grep -q '^profile = ' project_def.prop
  sed -i "s/^profile = .*/profile = ${TIZEN_PROFILE}/" project_def.prop
  grep -q '^USER_LIBS =' project_def.prop
  sed -i 's/^USER_LIBS =\(.*\)/USER_LIBS =\1 capi-nnstreamer/' project_def.prop
  tizen build-native -a "${TIZEN_ARCH}" -c "${TIZEN_COMPILER}" -C Debug
  test -f "Debug/$(sed -n 's/^APPNAME = //p' project_def.prop)"
  echo "Tizen example app (${app_name}) build test: PASS"
fi
