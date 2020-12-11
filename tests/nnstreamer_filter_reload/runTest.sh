#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
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

if [ -z ${SO_EXT} ]; then
    SO_EXT="so"
fi

# This is compatible with SSAT (https://github.com/myungjoo/SSAT)
testInit $1

# NNStreamer and plugins path for test
PATH_TO_PLUGIN="../../build"
# This test is valid only with the tensor filter extension for tflite
if [[ -d $PATH_TO_PLUGIN ]]; then
    ini_path="${PATH_TO_PLUGIN}/ext/nnstreamer/tensor_filter"
    if [[ -d ${ini_path} ]]; then
        check=$(ls ${ini_path} | grep tensorflow1-lite.${SO_EXT})
        if [[ ! $check ]]; then
            echo "Cannot find tensorflow1-lite shared lib"
            report
            exit
        fi
    else
        echo "Cannot find ${ini_path}"
    fi
else
    ini_file="/etc/nnstreamer.ini"
    if [[ -f ${ini_file} ]]; then
        path=$(grep "^filters" ${ini_file})
        key=${path%=*}
        value=${path##*=}

        if [[ $key != "filters" ]]; then
            echo "String Error"
            report
            exit
        fi

        if [[ -d ${value} ]]; then
            check=$(ls ${value} | grep tensorflow1-lite.${SO_EXT})
            if [[ ! $check ]]; then
                echo "Cannot find tensorflow1-lite shared lib"
                report
                exit
            fi
        else
            echo "Cannot file ${value}"
            report
            exit
        fi
    else
        echo "Cannot identify nnstreamer.ini"
        report
        exit
    fi
fi

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

${TESTBINDIR}/nnstreamer_filter_reload/unittest_filter_reload --input_img=${PATH_TO_INPUT} --first_model=${PATH_TO_MODEL1} --second_model=${PATH_TO_MODEL1}
testResult $? 1 "reload tflite model case 1 (same model)" 0 1

${TESTBINDIR}/nnstreamer_filter_reload/unittest_filter_reload --input_img=${PATH_TO_INPUT} --first_model=${PATH_TO_MODEL1} --second_model=${PATH_TO_MODEL2}
testResult $? 2 "reload tflite model case 2 (diff model)" 0 1

report
