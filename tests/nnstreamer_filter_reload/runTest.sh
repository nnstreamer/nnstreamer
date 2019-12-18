#!/usr/bin/env bash
##
## @file runTest.sh
## @author Dongju Chae <dongju.chae@samsung.com>
## @date Dec 19 2019
## @brief SSAT Test Case for a tensor filter's runtime model updates
##
if [[ "$SSATAPILOADED" != "1" ]]; then
    SILENT=0
    INDEPENDENT=1
    search="ssat-api.sh"
    source $search
    printf "${Blue}Independent Mode${NC}
"
fi

# This is compatible with SSAT (https://github.com/myungjoo/SSAT)
testInit $1

# NNStreamer and plugins path for test
PATH_TO_PLUGIN="../../build"
PATH_TO_INPUT="../test_models/data/orange.png"
PATH_TO_MODEL1="../test_models/models/mobilenet_v1_1.0_224_quant.tflite"
PATH_TO_MODEL2="../test_models/models/mobilenet_v2_1.0_224_quant.tflite"

if [[ ! -z "${UNITTEST_DIR}" ]]; then
    TESTBINDIR="${UNITTEST_DIR}"
elif [ ! -d "${PATH_TO_PLUGIN}" ] && [ ! -d "${UNITTEST_DIR}" ]; then
    TESTBINDIR="/usr/lib/nnstreamer/unittest"
else
    TESTBINDIR="../../build/tests"
fi

${TESTBINDIR}/unittest_filter_reload --input_img=${PATH_TO_INPUT} --first_model=${PATH_TO_MODEL1} --second_model=${PATH_TO_MODEL1}
testResult $? 1 "reload tflite model case 1 (same model)" 0 1

${TESTBINDIR}/unittest_filter_reload --input_img=${PATH_TO_INPUT} --first_model=${PATH_TO_MODEL1} --second_model=${PATH_TO_MODEL2}
testResult $? 2 "reload tflite model case 2 (diff model)" 0 1

report
