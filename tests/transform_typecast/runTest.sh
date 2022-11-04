#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
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
    printf "${Blue}Independent Mode${NC}"
fi

# This is compatible with SSAT (https://github.com/myungjoo/SSAT)
testInit $1

PATH_TO_PLUGIN="../../build"

if [ "$SKIPGEN" == "YES" ]; then
    echo "Test Case Generation Skipped"
    sopath=$2
else
    echo "Test Case Generation Started"
    python3 ../nnstreamer_converter/generateGoldenTestResult.py 8
    sopath=$1
fi
convertBMP2PNG

# Test gst availability. (0)
gstTest "videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=280,height=40,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! filesink location=\"testcase.apitest.log\" sync=true" 0 0 0 $PERFORMANCE

# Test with small stream (1, 2)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=typecast option=uint32 ! filesink location=\"testcase01.typecast.log\" sync=true t. ! queue ! filesink location=\"testcase01.direct.log\" sync=true" 1 0 0 $PERFORMANCE
# uint8 -> uint32
python3 checkResult.py typecast testcase01.direct.log testcase01.typecast.log uint8 1 B uint32 4 I
testResult $? 1 "Golden test comparison" 0 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=BGRx ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=typecast option=uint16 ! filesink location=\"testcase02.typecast.log\" sync=true t. ! queue ! filesink location=\"testcase02.direct.log\" sync=true" 2 0 0 $PERFORMANCE
# uint8 -> uint32
python3 checkResult.py typecast testcase02.direct.log testcase02.typecast.log uint8 1 B uint16 2 H
testResult $? 2 "Golden test comparison" 0 1

# Fail Test: Unknown data type is given
# uint8 -> uint128
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=BGRx ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=typecast option=uint128 ! filesink location=\"testcase02-F.typecast.log\" sync=true t. ! queue ! filesink location=\"testcase02-F.direct.log\" sync=true" 2F_n 0 1 $PERFORMANCE

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=BGRx ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=typecast option=int8 ! filesink location=\"testcase03.typecast.log\" sync=true t. ! queue ! filesink location=\"testcase03.direct.log\" sync=true" 3 0 0 $PERFORMANCE
# uint8 -> int8
python3 checkResult.py typecast testcase03.direct.log testcase03.typecast.log uint8 1 B int8 1 b
testResult $? 3 "Golden test comparison" 0 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=BGRx ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=typecast option=uint32 ! tensor_transform mode=typecast option=uint8 ! filesink location=\"testcase04.typecast.log\" sync=true t. ! queue ! filesink location=\"testcase04.direct.log\" sync=true" 4 0 0 $PERFORMANCE
# uint8 -> uint32 -> uint8
python3 checkResult.py typecast testcase04.direct.log testcase04.typecast.log uint8 1 B uint8 1 B
testResult $? 4 "Golden test comparison" 0 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=BGRx ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=typecast option=float32 ! filesink location=\"testcase05.typecast.log\" sync=true t. ! queue ! filesink location=\"testcase05.direct.log\" sync=true" 5 0 0 $PERFORMANCE
# uint8 -> float32
python3 checkResult.py typecast testcase05.direct.log testcase05.typecast.log uint8 1 B float32 4 f
testResult $? 5 "Golden test comparison" 0 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=BGRx ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=typecast option=float64 ! filesink location=\"testcase06.typecast.log\" sync=true t. ! queue ! filesink location=\"testcase06.direct.log\" sync=true" 6 0 0 $PERFORMANCE
# uint8 -> float64
python3 checkResult.py typecast testcase06.direct.log testcase06.typecast.log uint8 1 B float64 8 d
testResult $? 6 "Golden test comparison" 0 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=BGRx ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=typecast option=int8 ! tensor_transform mode=typecast option=float32 ! tensor_transform mode=typecast option=float64 ! tensor_transform mode=typecast option=int64 ! tensor_transform mode=typecast option=uint8 ! filesink location=\"testcase07.typecast.log\" sync=true t. ! queue ! filesink location=\"testcase07.direct.log\" sync=true" 7 0 0 $PERFORMANCE
# uint8 -> int8 -> float32 -> float64 -> int64 -> uint8
python3 checkResult.py typecast testcase07.direct.log testcase07.typecast.log uint8 1 B uint8 1 B
testResult $? 7 "Golden test comparison" 0 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=BGRx ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=typecast option=int8 ! tensor_transform mode=typecast option=float32 ! tensor_transform mode=typecast option=float64 ! tensor_transform mode=typecast option=uint8 ! filesink location=\"testcase08.typecast.log\" sync=true t. ! queue ! filesink location=\"testcase08.direct.log\" sync=true" 8 0 0 $PERFORMANCE
# uint8 -> int8 -> float32 -> float64 -> uint8
python3 checkResult.py typecast testcase08.direct.log testcase08.typecast.log uint8 1 B uint8 1 B
testResult $? 8 "Golden test comparison" 0 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=BGRx ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=typecast option=int8 ! tensor_transform mode=typecast option=float32 ! tensor_transform mode=typecast option=uint16 ! tensor_transform mode=typecast option=uint8 ! filesink location=\"testcase09.typecast.log\" sync=true t. ! queue ! filesink location=\"testcase09.direct.log\" sync=true" 9 0 0 $PERFORMANCE
# uint8 -> int8 -> float32 -> uint16 -> uint8
python3 checkResult.py typecast testcase09.direct.log testcase09.typecast.log uint8 1 B uint8 1 B
testResult $? 9 "Golden test comparison" 0 1

# Tensor tensors stream
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! \
        pngdec ! videoconvert ! video/x-raw, format=BGRx ! tensor_converter ! tee name=t \
        t. ! queue ! filesink location=\"testcase_tensors.direct.log\" buffer-mode=unbuffered sync=false async=false \
        t. ! queue ! mux.sink_0 \
        t. ! queue ! mux.sink_1 \
        t. ! queue ! mux.sink_2 \
        tensor_mux name=mux ! tensor_transform mode=typecast option=float32 apply=0,1,2 ! tensor_demux name=demux \
        demux.src_0 ! queue ! filesink location=\"testcase_tensors_0.typecast.log\" buffer-mode=unbuffered sync=false async=false \
        demux.src_1 ! queue ! filesink location=\"testcase_tensors_1.typecast.log\" buffer-mode=unbuffered sync=false async=false \
        demux.src_2 ! queue ! filesink location=\"testcase_tensors_2.typecast.log\" buffer-mode=unbuffered sync=false async=false" 10 0 0 $PERFORMANCE
python3 checkResult.py typecast testcase_tensors.direct.log testcase_tensors_0.typecast.log uint8 1 B float32 4 f
testResult $? 10 "Golden test comparison 0" 0 1
python3 checkResult.py typecast testcase_tensors.direct.log testcase_tensors_1.typecast.log uint8 1 B float32 4 f
testResult $? 10 "Golden test comparison 1" 0 1
python3 checkResult.py typecast testcase_tensors.direct.log testcase_tensors_2.typecast.log uint8 1 B float32 4 f
testResult $? 10 "Golden test comparison 2" 0 1

# Tensor tensors stream with apply option
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! \
        pngdec ! videoconvert ! video/x-raw, format=BGRx ! tensor_converter ! tee name=t \
        t. ! queue ! filesink location=\"testcase_apply.direct.log\" buffer-mode=unbuffered sync=false async=false \
        t. ! queue ! mux.sink_0 \
        t. ! queue ! mux.sink_1 \
        t. ! queue ! mux.sink_2 \
        tensor_mux name=mux ! tensor_transform mode=typecast option=float32 apply=0,2 ! tensor_demux name=demux \
        demux.src_0 ! queue ! filesink location=\"testcase_apply_0.typecast.log\" buffer-mode=unbuffered sync=false async=false \
        demux.src_1 ! queue ! filesink location=\"testcase_apply_1.typecast.log\" buffer-mode=unbuffered sync=false async=false \
        demux.src_2 ! queue ! filesink location=\"testcase_apply_2.typecast.log\" buffer-mode=unbuffered sync=false async=false" 11 0 0 $PERFORMANCE
python3 checkResult.py typecast testcase_apply.direct.log testcase_apply_0.typecast.log uint8 1 B float32 4 f
testResult $? 11-1 "apply test comparison 11-1" 0 1
callCompareTest testcase_apply.direct.log testcase_apply_1.typecast.log 11-2 "apply test comparison 1-1" 1 0
python3 checkResult.py typecast testcase_apply.direct.log testcase_apply_2.typecast.log uint8 1 B float32 4 f
testResult $? 11-2 "apply test comparison 11-2" 0 1


F16MAYFAIL=1
if [ '${FLOAT16_SUPPORTED}' == '1' ]; then F16MAYFAIL=0; fi

## Float16 tests : Test with small stream (12, 13)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=arithmetic option=div:8 ! tensor_transform mode=typecast option=float16 ! tee name=t ! queue ! filesink location=\"testcase12.noop.log\" sync=true t. ! queue ! tensor_transform mode=arithmetic option=mul:1.0,add:1.0,add:-1.0 ! filesink location=\"testcase12.ops.log\" sync=true t. ! queue ! tensor_transform mode=arithmetic option=add:0.0001 ! filesink location=\"testcase13.ops.log\" sync=true t. ! queue ! tensor_transform mode=arithmetic option=add:0.1 ! filesink location=\"testcase14.ops.log\" sync=true" 12 $F16MAYFAIL 0 $PERFORMANCE

## need python >= 3.6
python3 checkResult.py typecast testcase12.noop.log testcase12.ops.log float16 2 e float16 2 e
testResult $? 12 "float16 values * 1.0 + 1.0 - 1.0 == original test" $F16MAYFAIL 1

python3 checkResult.py typecast testcase12.noop.log testcase13.ops.log float16 2 e float16 2 e
testResult $? 13 "float16 values + 0.0001 is less than original + 0.001 test" $F16MAYFAIL 1

python3 checkResult.py typecast testcase12.noop.log testcase14.ops.log float16 2 e float16 2 e
val=$?
if [ $val -gt 0 ]; then val=1; fi
testResult $val 14 "float16 values + 0.1 >= original + 0.01 test" $F16MAYFAIL 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! video/x-raw,format=RGB,width=4,height=4 ! tensor_converter ! other/tensors,num_tensors=1,types=float16 ! fakesink" 15_n 0 1 $PERFORMANCE

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! video/x-raw,format=RGB,width=4,height=4 ! tensor_converter ! tensor_transform mode=typecast option=float32 ! other/tensors,num_tensors=1,types=float16 ! fakesink" 16_n 0 1 $PERFORMANCE

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! video/x-raw,format=RGB,width=4,height=4 ! tensor_converter ! tensor_transform mode=typecast option=float64 ! other/tensors,num_tensors=1,types=float16 ! fakesink" 17_n 0 1 $PERFORMANCE

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! video/x-raw,format=RGB,width=4,height=4 ! tensor_converter ! tensor_transform mode=typecast option=float16 !  other/tensors,num_tensors=1,types=float32 ! fakesink" 18_n 0 1 $PERFORMANCE

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! video/x-raw,format=RGB,width=4,height=4 ! tensor_converter ! tensor_transform mode=typecast option=float16 ! other/tensors,num_tensors=1,types=uint8 ! fakesink" 19_n 0 1 $PERFORMANCE

# Dimension declaration test case
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! video/x-raw,format=RGB,width=4,height=4 ! tensor_converter ! tensor_transform mode=typecast option=float32 ! other/tensors,num_tensors=1,types=float32,dimensions=3:4:4 ! fakesink" 20-1 0 0 $PERFORMANCE

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! video/x-raw,format=RGB,width=4,height=4 ! tensor_converter ! tensor_transform mode=typecast option=float32 ! other/tensor,types=float32,dimensions=3:4:4 ! fakesink" 20-2 0 0 $PERFORMANCE

rm *.log *.bmp *.png *.golden *.raw *.dat

report
