#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author Yongjoo Ahn <yongjoo1.ahn@samsung.com>
## @date 27 Jul 2021
## @brief SSAT Test Cases for NNStreamer tensor_sparse_enc and tensor_sparse_dec
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

PATH_TO_PLUGIN="../../build"
# Check lua libraries are built. This test utilizes it for making "sparse" dense tensors.
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

# Make sample dense tensor (two random positions are non-zero values)
MAKE_SAMPLE_TENSORS_SCRIPT="
inputTensorsInfo = { num = 1, dim = {{3, 10, 10, 1},}, type = {'uint8',} }
outputTensorsInfo = { num = 1, dim = {{1, 3, 4, 1},}, type = {'uint8',} }


function nnstreamer_invoke()
  math.randomseed(os.time())
  oC = outputTensorsInfo['dim'][1][1]
  oW = outputTensorsInfo['dim'][1][2]
  oH = outputTensorsInfo['dim'][1][3]
  oN = outputTensorsInfo['dim'][1][4]

  output = output_tensor(1)

  ww = math.random(oW)
  hh = math.random(oH)
  outIndex = 0
  outIndex = outIndex + (hh - 1)*oW*oC
  outIndex = outIndex + (ww - 1)*oC
  outIndex = outIndex + 1
  output[outIndex] = math.random(127)

  ww = math.random(oW)
  hh = math.random(oH)
  outIndex = 0
  outIndex = outIndex + (hh - 1)*oW*oC
  outIndex = outIndex + (ww - 1)*oC
  outIndex = outIndex + 1
  output[outIndex] = math.random(127)

end
"

# Make sample dense tensors (2 tensors, two random positions of each tensor are non-zero values)
MAKE_SAMPLE_2TENSORS_SCRIPT="
inputTensorsInfo = { num = 1, dim = {{3, 10, 10, 1},}, type = {'uint8',} }
outputTensorsInfo = { num = 2, dim = {{1, 3, 4, 1}, {1, 5, 5, 1}}, type = {'uint8', 'float32'} }


function nnstreamer_invoke()
  oC = outputTensorsInfo['dim'][1][1]
  oW = outputTensorsInfo['dim'][1][2]
  oH = outputTensorsInfo['dim'][1][3]
  oN = outputTensorsInfo['dim'][1][4]
  output = output_tensor(1)

  ww = math.random(oW)
  hh = math.random(oH)
  outIndex = 0
  outIndex = outIndex + (hh - 1)*oW*oC
  outIndex = outIndex + (ww - 1)*oC
  outIndex = outIndex + 1
  output[outIndex] = math.random(127)

  ww = math.random(oW)
  hh = math.random(oH)
  outIndex = 0
  outIndex = outIndex + (hh - 1)*oW*oC
  outIndex = outIndex + (ww - 1)*oC
  outIndex = outIndex + 1
  output[outIndex] = math.random(127)

  oC = outputTensorsInfo['dim'][2][1]
  oW = outputTensorsInfo['dim'][2][2]
  oH = outputTensorsInfo['dim'][2][3]
  oN = outputTensorsInfo['dim'][2][4]
  output = output_tensor(2)

  ww = math.random(oW)
  hh = math.random(oH)
  outIndex = 0
  outIndex = outIndex + (hh - 1)*oW*oC
  outIndex = outIndex + (ww - 1)*oC
  outIndex = outIndex + 1
  output[outIndex] = math.random(127)

  ww = math.random(oW)
  hh = math.random(oH)
  outIndex = 0
  outIndex = outIndex + (hh - 1)*oW*oC
  outIndex = outIndex + (ww - 1)*oC
  outIndex = outIndex + 1
  output[outIndex] = math.random(127)

end
"

# Test encoding and decoding
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
videotestsrc num-buffers=1 ! \
    video/x-raw,format=RGB,width=10,height=10,framerate=0/1 ! videoconvert ! \
    tensor_converter ! tensor_filter framework=lua \
    model=\"${MAKE_SAMPLE_TENSORS_SCRIPT}\" ! tee name=t \
        t. ! queue ! filesink location=sample1.dense sync=true \
        t. ! queue ! tensor_sparse_enc ! \
            other/tensors,format=sparse,framerate=0/1 ! \
            tensor_sparse_dec ! \
            filesink location=dec1.result sync=true
" 1 0 0 $PERFORMANCE
callCompareTest sample1.dense dec1.result 1-1 "Compare 1" 0 0

# Should set types and framerate in capsfilter
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    filesrc location=sample1.dense ! \
    other/tensors,num_tensors=1,dimensions=1:3:4:1 ! \
    tensor_sparse_enc ! \
    tensor_sink" 1_n 0 1 $PERFORMANCE

# Test encoding and decoding with `num_tensors=2`
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
videotestsrc num-buffers=1 ! \
    video/x-raw,format=RGB,width=10,height=10,framerate=0/1 ! videoconvert ! \
    tensor_converter ! tensor_filter framework=lua \
    model=\"${MAKE_SAMPLE_2TENSORS_SCRIPT}\" ! tee name=t \
    t. ! queue ! filesink location=sample2.dense sync=true \
    t. ! queue ! tensor_sparse_enc ! \
    other/tensors,format=sparse,framerate=0/1 ! \
    tensor_sparse_dec ! \
    filesink location=dec2.result sync=true
" 2 0 0 $PERFORMANCE
callCompareTest sample2.dense dec2.result 2-1 "Compare 2" 0 0

# Test with tensor_converter
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
filesrc location=sample1.dense ! \
    application/octet-stream ! \
    tensor_converter input-dim=1:3:4:1 input-type=uint8 ! \
    tensor_sparse_enc ! tensor_sink" 3 0 0 $PERFORMANCE

# Sparse tensor filesrc/sink test (for `num_tensors=1`)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
videotestsrc num-buffers=1 ! \
    video/x-raw,format=RGB,width=10,height=10,framerate=0/1 ! videoconvert ! \
    tensor_converter ! tensor_filter framework=lua \
    model=\"${MAKE_SAMPLE_TENSORS_SCRIPT}\" ! tensor_sparse_enc ! filesink location=./sample1.sparse" 4 0 0 $PERFORMANCE

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
filesrc location=sample1.sparse ! \
    other/tensors,format=sparse,framerate=0/1 ! \
    tensor_sparse_dec ! \
    other/tensors,num_tensors=1,framerate=0/1,dimensions=1:3:4:1,types=uint8 ! \
    tensor_sparse_enc ! \
    filesink location=enc1.result sync=true" 5 0 0 $PERFORMANCE
callCompareTest sample1.sparse enc1.result 5-1 "Compare 5" 0 0

DEC_RESULT_TEST_SCRIPT="
inputTensorsInfo = {
    num = 1,
    dim = {{1, 3, 4, 1},},
    type = {'uint8',}
}

outputTensorsInfo = {
    num = 1,
    dim = {{1, 3, 4, 1},},
    type = {'uint8',}
}

function nnstreamer_invoke()
    input = input_tensor(1)
    output = output_tensor(1)
    for i=1,1*4*3*1 do
        output[i] = input[i] --[[ copy input into output --]]
    end
end
"

# Test with tensor_transform and tensor_filter
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
filesrc location=sample1.sparse ! \
    other/tensors,format=sparse,framerate=0/1 ! \
    tensor_sparse_dec ! \
    tensor_transform mode=arithmetic option=add:1,div:1 ! \
    tensor_filter framework=lua model=\"${DEC_RESULT_TEST_SCRIPT}\" ! \
    tensor_sink" 6 0 0 $PERFORMANCE

rm *.dense *.result *.sparse

report
