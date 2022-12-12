#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author Yongjoo Ahn <yongjoo1.ahn@samsung.com>
## @date Jun 3 2021
## @brief SSAT Test Cases for NNStreamer tensor_filter::lua
##

if [[ "$SSATAPILOADED" != "1" ]]; then
    SILENT=0
    INDEPENDENT=1
    search="ssat-api.sh"
    source $search
    printf "${Blue}Independent Mode${NC}"
fi

# This is compatible with SSAT (https://github.com/myungjoo/SSAT)
testInit $1

PATH_TO_PLUGIN="../../build"
# Check lua libraies are built
if [[ -d $PATH_TO_PLUGIN ]]; then
    ini_path="${PATH_TO_PLUGIN}/ext/nnstreamer/tensor_filter"
    if [[ -d ${ini_path} ]]; then
        check=$(ls ${ini_path} | grep lua.so)
        if [[ ! $check ]]; then
            echo "Cannot find lua shared lib"
            report
            exit
        fi
    else
        echo "Cannot find ${ini_path}"
        report exit
    fi
else
    echo "No build directory"
    report
    exit
fi

PATH_TO_PASSTHROUGH_MODEL="../test_models/models/passthrough.lua"
PATH_TO_SCALER_MODEL="../test_models/models/scaler.lua"

# Passthrough test
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=640,height=480,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=lua model=\"${PATH_TO_PASSTHROUGH_MODEL}\" ! filesink location=\"testcase1.passthrough.log\" sync=true t. ! queue ! filesink location=\"testcase1.direct.log\" sync=true" 1 0 0 $PERFORMANCE
callCompareTest testcase1.direct.log testcase1.passthrough.log 1 "Compare 1" 0 0

# Negative test for wrong input. Input should be 3 : 640 : 480 : 1
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=643,height=482,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_filter framework=lua model=\"${PATH_TO_PASSTHROUGH_MODEL}\" ! fakesink" 1_n 0 1 $PERFORMANCE

# Passthrough -> Scaler test
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=640,height=480,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=lua model=\"${PATH_TO_PASSTHROUGH_MODEL}\" ! tensor_filter framework=lua model=\"${PATH_TO_SCALER_MODEL}\" ! filesink location=\"testcase2.scaled.log\" sync=true t. ! queue ! filesink location=\"testcase2.direct.log\" sync=true" 2 0 0 $PERFORMANCE
python3 ../nnstreamer_filter_python3/checkScaledTensor.py testcase2.direct.log 640 480 testcase2.scaled.log 320 240 3
testResult $? 2 "Golden test comparison" 0 1

# Passthrough test with given string script
PASSTHROUGH_SCRIPT="
inputTensorsInfo = {
    num = 1,
    dim = {{3, 299, 299, 1},},
    type = {'uint8',} --[[ USE single quote to declare string --]]
}

outputTensorsInfo = {
    num = 1,
    dim = {{3, 299, 299, 1},},
    type = {'uint8',} --[[ USE single quote to declare string --]]
}

function nnstreamer_invoke()
    input = input_tensor(1)
    output = output_tensor(1)
    for i=1,3*299*299 do
        output[i] = input[i] --[[ copy input into output --]]
    end
end
"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=299,height=299,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=lua model=\"${PASSTHROUGH_SCRIPT}\" ! filesink location=\"testcase3.passthrough.log\" sync=true t. ! queue ! filesink location=\"testcase3.direct.log\" sync=true" 3 0 0 $PERFORMANCE
callCompareTest testcase3.direct.log testcase3.passthrough.log 1 "Compare 2" 0 0

PASSTHROUGH_INVALID_SCRIPT_SINGLE_QUOTE="
--[[ DO NOT USE double quote in Lua script --]]
inputTensorsInfo = {
    num = 1,
    dim = {{3, 299, 299, 1},},
    type = {\"uint8\",}
}

outputTensorsInfo = {
    num = 1,
    dim = {{3, 299, 299, 1},},
    type = {\"uint8\",}
}

function nnstreamer_invoke()
    input = input_tensor(1)
    output = output_tensor(1)
    for i=1,3*299*299 do
        output[i] = input[i]
    end
end
"
# Negative test for wrong Lua script. Should use single quote for string
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=299,height=299,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_filter framework=lua model=\"${PASSTHROUGH_INVALID_SCRIPT_SINGLE_QUOTE}\" ! fakesink" 3_n 0 1 $PERFORMANCE

# In script mode do not use '--' to comment. use '--[[ ... --]]'
PASSTHROUGH_INVALID_SCRIPT_COMMENT_FORMAT="
inputTensorsInfo = {
    num = 1,
    dim = {{3, 299, 299, 1},},
    type = {'uint8',}
}

outputTensorsInfo = {
    num = 1,
    dim = {{3, 299, 299, 1},},
    type = {'uint8',}
}

function nnstreamer_invoke()
    input = input_tensor(1)
    output = output_tensor(1)
    for i=1,3*299*299 do
        output[i] = input[i] -- copy input into output. DO NOT use double dashes for commenting
    end
end
"
# Negative test for wrong Lua script. Should use --[[ COMMENT --]] for comment
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=299,height=299,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_filter framework=lua model=\"${PASSTHROUGH_INVALID_SCRIPT_COMMENT_FORMAT}\" ! fakesink" 4_n 0 1 $PERFORMANCE

PASSTHROUGH_SCRIPT_F16="
inputTensorsInfo = {
    num = 1,
    dim = {{3, 299, 299, 1},},
    type = {'float16',} --[[ USE single quote to declare string --]]
}

outputTensorsInfo = {
    num = 1,
    dim = {{3, 299, 299, 1},},
    type = {'float16',} --[[ USE single quote to declare string --]]
}

function nnstreamer_invoke()
    input = input_tensor(1)
    output = output_tensor(1)
    for i=1,3*299*299 do
        output[i] = input[i] --[[ copy input into output --]]
    end
end
"
PASSTHROUGH_SCRIPT_F32="
inputTensorsInfo = {
    num = 1,
    dim = {{3, 299, 299, 1},},
    type = {'float32',} --[[ USE single quote to declare string --]]
}

outputTensorsInfo = {
    num = 1,
    dim = {{3, 299, 299, 1},},
    type = {'float32',} --[[ USE single quote to declare string --]]
}

function nnstreamer_invoke()
    input = input_tensor(1)
    output = output_tensor(1)
    for i=1,3*299*299 do
        output[i] = input[i] --[[ copy input into output --]]
    end
end
"

# LUA-Float16
F16MAYFAIL=1
if [ '${FLOAT16_SUPPORTED}' == '1' ]; then F16MAYFAIL=0; fi

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=299,height=299,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=typecast option=float16 ! tensor_filter framework=lua model=\"${PASSTHROUGH_SCRIPT_F16}\" ! fakesink" 5 $F16MAYFAIL 0 $PERFORMANCE
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=299,height=299,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=typecast option=float32 ! tensor_filter framework=lua model=\"${PASSTHROUGH_SCRIPT_F32}\" ! fakesink" 6 0 0 $PERFORMANCE
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=299,height=299,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=typecast option=float16 ! tensor_filter framework=lua model=\"${PASSTHROUGH_SCRIPT_F32}\" ! fakesink" 7_n 0 1 $PERFORMANCE
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=299,height=299,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=typecast option=float32 ! tensor_filter framework=lua model=\"${PASSTHROUGH_SCRIPT_F16}\" ! fakesink" 8_n 0 1 $PERFORMANCE
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=299,height=299,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=typecast option=float16 ! tensor_filter framework=lua model=\"${PASSTHROUGH_SCRIPT}\" ! fakesink" 9_n 0 1 $PERFORMANCE
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=299,height=299,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=typecast option=float32 ! tensor_filter framework=lua model=\"${PASSTHROUGH_SCRIPT_F16}\" ! fakesink" 10_n 0 1 $PERFORMANCE
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=299,height=299,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=typecast option=float64 ! tensor_filter framework=lua model=\"${PASSTHROUGH_SCRIPT_F16}\" ! fakesink" 11_n 0 1 $PERFORMANCE

# Passthrough test with given string script
# Not declaring full dimension
PASSTHROUGH_SCRIPT="
inputTensorsInfo = {
    num = 1,
    dim = {{3, 299, 299},},
    type = {'uint8',} --[[ USE single quote to declare string --]]
}

outputTensorsInfo = {
    num = 1,
    dim = {{3, 299, 299},},
    type = {'uint8',} --[[ USE single quote to declare string --]]
}

function nnstreamer_invoke()
    input = input_tensor(1)
    output = output_tensor(1)
    for i=1,3*299*299 do
        output[i] = input[i] --[[ copy input into output --]]
    end
end
"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=299,height=299,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=lua model=\"${PASSTHROUGH_SCRIPT}\" ! filesink location=\"testcase12.passthrough.log\" sync=true t. ! queue ! filesink location=\"testcase12.direct.log\" sync=true" 12 0 0 $PERFORMANCE
callCompareTest testcase12.direct.log testcase12.passthrough.log 12 "Compare 3" 0 0


rm *.log

report
