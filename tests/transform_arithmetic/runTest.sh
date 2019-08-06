#!/usr/bin/env bash
##
## @file runTest.sh
## @author MyungJoo Ham <myungjoo.ham@gmail.com>
## @date Nov 01 2018
## @brief SSAT Test Cases for NNStreamer
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

if [ "$SKIPGEN" == "YES" ]; then
    echo "Test Case Generation Skipped"
    sopath=$2
else
    echo "Test Case Generation Started"
    python ../nnstreamer_converter/generateGoldenTestResult.py 8
    sopath=$1
fi
convertBMP2PNG

# Test gst availability. (0)
gstTest "videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=280,height=40,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! filesink location=\"testcase.apitest.log\" sync=true" 0 0 0 $PERFORMANCE

# Test with small stream (1, 2)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=typecast option=float32 ! tee name=t ! queue ! tensor_transform mode=arithmetic option=add:-10 ! filesink location=\"testcase01.arithmetic.log\" sync=true t. ! queue ! filesink location=\"testcase01.direct.log\" sync=true" 1 0 0 $PERFORMANCE

python checkResult.py arithmetic testcase01.direct.log testcase01.arithmetic.log 4 4 f f add -10 0
testResult $? 1 "Golden test comparison" 0 1

# Test with small stream (1, 2)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=typecast option=float32 ! tee name=t ! queue ! tensor_transform mode=arithmetic option=mul:2.0 ! filesink location=\"testcase02.arithmetic.log\" sync=true t. ! queue ! filesink location=\"testcase02.direct.log\" sync=true" 2 0 0 $PERFORMANCE

python checkResult.py arithmetic testcase02.direct.log testcase02.arithmetic.log 4 4 f f mul 2.0 0
testResult $? 2 "Golden test comparison" 0 1

# Test for mul with floating-point operand
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=typecast option=float32 ! tee name=t ! queue ! tensor_transform mode=arithmetic option=mul:-5.5 ! filesink location=\"testcase03.arithmetic.1.log\" sync=true t. ! queue ! filesink location=\"testcase03.direct.1.log\" sync=true" 3 0 0 $PERFORMANCE

python checkResult.py arithmetic testcase03.direct.1.log testcase03.arithmetic.1.log 4 4 f f mul -5.5 0
testResult $? 3 "Golden test comparison" 0 1

# Test 3-2 for typecast,mul with floating-point operand
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=arithmetic option=typecast:float32,mul:-5.5 ! filesink location=\"testcase03.arithmetic.2.log\" sync=true t. ! queue ! tensor_transform mode=typecast option=float32 ! filesink location=\"testcase03.direct.2.log\" sync=true" 3-2 0 0 $PERFORMANCE

python checkResult.py arithmetic testcase03.direct.2.log testcase03.arithmetic.2.log 4 4 f f mul -5.5 0
testResult $? 3-2 "Golden test comparison" 0 1

# Test for add with floating-point operand
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=typecast option=float64 ! tee name=t ! queue ! tensor_transform mode=arithmetic option=add:9.900000e-001 ! filesink location=\"testcase04.arithmetic.log\" sync=true t. ! queue ! filesink location=\"testcase04.direct.log\" sync=true" 4 0 0 $PERFORMANCE

python checkResult.py arithmetic testcase04.direct.log testcase04.arithmetic.log 8 8 d d add 9.900000e-001 0
testResult $? 4 "Golden test comparison" 0 1

# Test for add with two floating-point operand: the second operand will be ignored.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=typecast option=float64 ! tee name=t ! queue ! tensor_transform mode=arithmetic option=add:9.900000e-001:-80.256 ! filesink location=\"testcase04.arithmetic.ok.log\" sync=true t. ! queue ! filesink location=\"testcase04.direct.ok.log\" sync=true" 4-OK 0 0 $PERFORMANCE

python checkResult.py arithmetic testcase04.direct.ok.log testcase04.arithmetic.ok.log 8 8 d d add 9.900000e-001 -80.256
testResult $? 4-OK "Golden test comparison" 0 1

# Test for add,mul with floating-point operands
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=typecast option=float64 ! tee name=t ! queue ! tensor_transform mode=arithmetic option=add:-9.3,mul:-11.4823e-002 ! filesink location=\"testcase05.arithmetic.log\" sync=true t. ! queue ! filesink location=\"testcase05.direct.log\" sync=true" 5 0 0 $PERFORMANCE

testResult $? 5 "Golden test comparison" 0 1
python checkResult.py arithmetic testcase05.direct.log testcase05.arithmetic.log 8 8 d d add-mul -9.3 -11.4823e-002

# Test for mul,add with floating-point operands
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=typecast option=float64 ! tee name=t ! queue ! tensor_transform mode=arithmetic option=mul:-50.0987e+003,add:15.3 ! filesink location=\"testcase06.arithmetic.log\" sync=true t. ! queue ! filesink location=\"testcase06.direct.log\" sync=true" 6 0 0 $PERFORMANCE

testResult $? 6 "Golden test comparison" 0 1
python checkResult.py arithmetic testcase06.direct.log testcase06.arithmetic.log 8 8 d d add-mul -50.0987e+003 15.3

# Test for mul with tensors typecasted to int8 (acceleration=false)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=arithmetic option=typecast:int8,mul:-3 acceleration=false ! filesink location=\"testcase07.arithmetic.1.log\" sync=true t. ! queue ! tensor_transform mode=typecast option=int8 acceleration=false ! filesink location=\"testcase07.direct.1.log\" sync=true" 7-1 0 0 $PERFORMANCE

python checkResult.py arithmetic testcase07.direct.1.log testcase07.arithmetic.1.log 1 1 b b mul -3 0
testResult $? 7-1 "Golden test comparison" 0 1

# Test for mul with tensors typecasted to uint8 (acceleration=false)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=arithmetic option=typecast:uint8,mul:1 acceleration=false ! filesink location=\"testcase07.arithmetic.2.log\" sync=true t. ! queue ! tensor_transform mode=typecast option=uint8 acceleration=false ! filesink location=\"testcase07.direct.2.log\" sync=true" 7-2 0 0 $PERFORMANCE

python checkResult.py arithmetic testcase07.direct.2.log testcase07.arithmetic.2.log 1 1 B B mul 1 0
testResult $? 7-2 "Golden test comparison" 0 1

# Test for mul with tensors typecasted to int16 (acceleration=false)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=arithmetic option=typecast:int16,mul:-16 acceleration=false ! filesink location=\"testcase07.arithmetic.3.log\" sync=true t. ! queue ! tensor_transform mode=typecast option=int16 acceleration=false ! filesink location=\"testcase07.direct.3.log\" sync=true" 7-3 0 0 $PERFORMANCE

python checkResult.py arithmetic testcase07.direct.3.log testcase07.arithmetic.3.log 2 2 h h mul -16 0
testResult $? 7-3 "Golden test comparison" 0 1

# Test for mul with tensors typecasted to uint16 (acceleration=false)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=arithmetic option=typecast:uint16,mul:16 acceleration=false ! filesink location=\"testcase07.arithmetic.4.log\" sync=true t. ! queue ! tensor_transform mode=typecast option=uint16 acceleration=false ! filesink location=\"testcase07.direct.4.log\" sync=true" 7-4 0 0 $PERFORMANCE

python checkResult.py arithmetic testcase07.direct.4.log testcase07.arithmetic.4.log 2 2 H H mul 16 0
testResult $? 7-4 "Golden test comparison" 0 1

# Test for mul with tensors typecasted to int32 (acceleration=false)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=arithmetic option=typecast:int32,mul:-255 acceleration=false ! filesink location=\"testcase07.arithmetic.5.log\" sync=true t. ! queue ! tensor_transform mode=typecast option=int32 acceleration=false ! filesink location=\"testcase07.direct.5.log\" sync=true" 7-5 0 0 $PERFORMANCE

python checkResult.py arithmetic testcase07.direct.5.log testcase07.arithmetic.5.log 4 4 i i mul -255 0
testResult $? 7-5 "Golden test comparison" 0 1

# Test for mul with tensors typecasted to uint32 (acceleration=false)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=arithmetic option=typecast:uint32,mul:255 acceleration=false ! filesink location=\"testcase07.arithmetic.6.log\" sync=true t. ! queue ! tensor_transform mode=typecast option=uint32 acceleration=false ! filesink location=\"testcase07.direct.6.log\" sync=true" 7-6 0 0 $PERFORMANCE

python checkResult.py arithmetic testcase07.direct.6.log testcase07.arithmetic.6.log 4 4 I I mul 255 0
testResult $? 7-6 "Golden test comparison" 0 1

# Test for mul with tensors typecasted to int64 (acceleration=false)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=arithmetic option=typecast:int64,mul:-65535 acceleration=false ! filesink location=\"testcase07.arithmetic.7.log\" sync=true t. ! queue ! tensor_transform mode=typecast option=int64 acceleration=false ! filesink location=\"testcase07.direct.7.log\" sync=true" 7-7 0 0 $PERFORMANCE

python checkResult.py arithmetic testcase07.direct.7.log testcase07.arithmetic.7.log 8 8 q q mul -65535 0
testResult $? 7-7 "Golden test comparison" 0 1

# Test for mul with tensors typecasted to uint64 (acceleration=false)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=arithmetic option=typecast:uint64,mul:65535 acceleration=false ! filesink location=\"testcase07.arithmetic.8.log\" sync=true t. ! queue ! tensor_transform mode=typecast option=uint64 acceleration=false ! filesink location=\"testcase07.direct.8.log\" sync=true" 7-8 0 0 $PERFORMANCE

python checkResult.py arithmetic testcase07.direct.8.log testcase07.arithmetic.8.log 8 8 Q Q mul 65535 0
testResult $? 7-8 "Golden test comparison" 0 1

# Fail Test for the option string is wrong
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=arithmetic option=casttype:uint64,mul:65535 acceleration=false ! fakesink sync=true " 8F_n 0 1 $PERFORMANCE

report
