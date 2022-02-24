#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author MyungJoo Ham <myungjoo.ham@gmail.com>
## @date Mar 06 2019
## @brief SSAT Test Cases for NNStreamer/Codegen
##
if [[ "$SSATAPILOADED" != "1" ]]; then
    SILENT=0
    INDEPENDENT=1
    search="ssat-api.sh"
    source $search
    printf "${Blue}Independent Mode${NC}
"
fi

if [ -z ${SO_EXT} ]; then
    SO_EXT="so"
fi

# For macOS support
if [[ ${SO_EXT} == 'dylib' ]]; then
    EXTRA_PKG_CONFIG=/usr/local/opt/libffi/lib/pkgconfig
fi

# This is compatible with SSAT (https://github.com/myungjoo/SSAT)
testInit $1

PATH_TO_PLUGIN="../../build"
pwd=$(pwd)
rm -f nnstreamer-single.pc
rm -f nnstreamer.pc

cat <<EOF >nnstreamer-single.pc
Name: nnstreamer-single
Version: 2.1.0
Description: temporary nnstreamer-single pkgconfig for unittesting during build
Requires:
Libs: -L${pwd}/../../build/gst/nnstreamer -lnnstreamer-single
Cflags: -I${pwd}/../../gst/nnstreamer -I${pwd}/../../gst/nnstreamer/include -I${pwd}/${PATH_TO_PLUGIN}/gst/nnstreamer/include
EOF

cat <<EOF >nnstreamer.pc
Name: nnstreamer
Description: temporary nnstreamer pkgconfig for unittesting during build
Version: 0.1.2
Requires: nnstreamer-single
Libs: -L${pwd}/../../build/gst/nnstreamer -lnnstreamer
Cflags: -I${pwd}/../../gst/nnstreamer -I${pwd}/../../gst/nnstreamer/include -I${pwd}/${PATH_TO_PLUGIN}/gst/nnstreamer/include
EOF

function do_test() {
    rm -f tc${1}.c
    rm -f meson.build

    cat <<EOF | python3 ../../tools/development/nnstreamerCodeGenCustomFilter.py
testcase${1}
tc${1}
${2}
${3}
EOF

    testResult $? TC${1} "CodeGen of ${2}/${3} case" 0 1

    rm -rf build${1}
    PKG_CONFIG_PATH=.:$EXTRA_PKG_CONFIG:$PKG_CONFIG_PATH meson build${1} && ninja -C build${1}
    testResult $? TC${1}C "Build codegen result of TC1 (${2}/${3} case)" 0 1

    gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num_buffers=10 ! videoconvert ! videoscale ! video/x-raw,width=224,height=224,format=RGB ! tensor_converter ! tensor_filter framework=custom model=\"build${1}/libtc${1}.${SO_EXT}\" ! fakesink" TC${1}R "Run gst-launch for TC${1}" 0 0 $PERFORMANCE
}

do_test 01 y y
do_test 02 y n
do_test 03 n y
do_test 04 n n

report
