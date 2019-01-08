#!/usr/bin/env bash
##
## @file runTest.sh
## @author MyungJoo Ham <myungjoo.ham@gmail.com>
## @date Nov 01 2018
## @brief SSAT Test Cases for NNStreamer
##
if [[ "$SSATAPILOADED" != "1" ]]
then
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

if [ "$SKIPGEN" == "YES" ]
then
  echo "Test Case Generation Skipped"
else
  echo "Test Case Generation Started"
  python ../nnstreamer_converter/generateGoldenTestResult.py 8
fi
convertBMP2PNG

# Test gst availability. (0)
gstTest "videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=280,height=40,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! filesink location=\"testcase02.apitest.log\" sync=true" 0 0 0 $PERFORMANCE


# Test constant passthrough custom filter (1, 2)
PATH_TO_MODEL="../../build/nnstreamer_example/custom_example_passthrough/libnnstreamer_customfilter_passthrough.so"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=280,height=40,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=\"custom\" model=\"${PATH_TO_MODEL}\" input=\"3:280:40\" inputtype=\"uint8\" output=\"3:280:40\" outputtype=\"uint8\" ! filesink location=\"testcase01.passthrough.log\" sync=true t. ! queue ! filesink location=\"testcase01.direct.log\" sync=true" 1 0 0 $PERFORMANCE

callCompareTest testcase01.direct.log testcase01.passthrough.log 1 "Compare 1" 0 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=280,height=40,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=\"custom\" model=\"${PATH_TO_MODEL}\" ! filesink location=\"testcase02.passthrough.log\" sync=true t. ! queue ! filesink location=\"testcase02.direct.log\" sync=true" 2 0 0 $PERFORMANCE

callCompareTest testcase02.direct.log testcase02.passthrough.log 2 "Compare 2" 0 0


# Test variable-dim passthrough custom filter (3, 4)
PATH_TO_MODEL_V="../../build/nnstreamer_example/custom_example_passthrough/libnnstreamer_customfilter_passthrough_variable.so"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=640,height=480,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=\"custom\" model=\"${PATH_TO_MODEL_V}\" input=\"3:640:480\" inputtype=\"uint8\" output=\"3:640:480\" outputtype=\"uint8\" ! filesink location=\"testcase03.passthrough.log\" sync=true t. ! queue ! filesink location=\"testcase03.direct.log\" sync=true" 3 0 0 $PERFORMANCE

callCompareTest testcase03.direct.log testcase03.passthrough.log 3 "Compare 3" 0 0

# Test single tensor (other/tensor)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=640,height=480,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=\"custom\" model=\"${PATH_TO_MODEL_V}\" ! filesink location=\"testcase04.tensor.passthrough.log\" sync=true t. ! queue ! filesink location=\"testcase04.tensor.direct.log\" sync=true" 4-1 0 0 $PERFORMANCE

callCompareTest testcase04.tensor.direct.log testcase04.tensor.passthrough.log 4-2 "Compare 4-2" 0 0

# Test multi tensors with mux (other/tensors)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_mux name=mux ! tee name=t ! queue ! tensor_filter framework=\"custom\" model=\"${PATH_TO_MODEL_V}\" ! filesink location=\"testcase04.tensors.passthrough.log\" sync=true t. ! queue ! filesink location=\"testcase04.tensors.direct.log\" sync=true videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=160,height=120,framerate=0/1 ! tensor_converter ! mux.sink_0 videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=120,height=80,framerate=0/1 ! tensor_converter ! mux.sink_1 videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=64,height=48,framerate=0/1 ! tensor_converter ! mux.sink_2" 4-3 0 0 $PERFORMANCE

callCompareTest testcase04.tensors.direct.log testcase04.tensors.passthrough.log 4-4 "Compare 4-4" 0 0

# Test scaler (5, 6, 7)
PATH_TO_MODEL_S="../../build/nnstreamer_example/custom_example_scaler/libnnstreamer_customfilter_scaler.so"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=640,height=480,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=\"custom\" model=\"${PATH_TO_MODEL_S}\" ! filesink location=\"testcase05.passthrough.log\" sync=true t. ! queue ! filesink location=\"testcase05.direct.log\" sync=true" 5 0 0 $PERFORMANCE
callCompareTest testcase05.direct.log testcase05.passthrough.log 5 "Compare 5" 0 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=640,height=480,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=\"custom\" model=\"${PATH_TO_MODEL_S}\" custom=\"320x240\" ! filesink location=\"testcase06.scaled.log\" sync=true t. ! queue ! filesink location=\"testcase06.direct.log\" sync=true" 6 0 0 $PERFORMANCE
python checkScaledTensor.py testcase06.direct.log 640 480 testcase06.scaled.log 320 240 3
testResult $? 6 "Golden test comparison" 0 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=640,height=480,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=\"custom\" model=\"${PATH_TO_MODEL_S}\" custom=\"1280X960\" ! filesink location=\"testcase07.scaled.log\" sync=true t. ! queue ! filesink location=\"testcase07.direct.log\" sync=true" 6 0 0 $PERFORMANCE
python checkScaledTensor.py testcase07.direct.log 640 480 testcase07.scaled.log 1280 960 3
testResult $? 7 "Golden test comparison" 0 1

# Test average (8)
PATH_TO_MODEL_A="../../build/nnstreamer_example/custom_example_average/libnnstreamer_customfilter_average.so"
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=640,height=480,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=\"custom\" model=\"${PATH_TO_MODEL_A}\" ! filesink location=\"testcase08.average.log\" sync=true t. ! queue ! filesink location=\"testcase08.direct.log\" sync=true" 8 0 0 $PERFORMANCE

# Apply average to stream (9)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=\"custom\" model=\"${PATH_TO_MODEL_A}\" ! filesink location=\"testcase09.average.log\" sync=true t. ! queue ! filesink location=\"testcase09.direct.log\" sync=true" 9 0 0 $PERFORMANCE


# Apply scaler to stream (10)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=\"custom\" model=\"${PATH_TO_MODEL_S}\" custom=\"8x8\" ! filesink location=\"testcase10.average.log\" sync=true t. ! queue ! filesink location=\"testcase10.direct.log\" sync=true" 10 0 0 $PERFORMANCE


# Test scaler + in-invoke allocator (11)
PATH_TO_MODEL_SI="../../build/nnstreamer_example/custom_example_scaler/libnnstreamer_customfilter_scaler_allocator.so"
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=640,height=480,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=\"custom\" model=\"${PATH_TO_MODEL_SI}\" custom=\"320x240\" ! filesink location=\"testcase11.scaled.log\" sync=true t. ! queue ! filesink location=\"testcase11.direct.log\" sync=true" 11 0 0 $PERFORMANCE
python checkScaledTensor.py testcase11.direct.log 640 480 testcase11.scaled.log 320 240 3
testResult $? 11 "Golden test comparison" 0 1

# OpenCV Test
# Test scaler using OpenCV (12, 13, 14)
PATH_TO_MODEL="../../build/nnstreamer_example/custom_example_opencv/libnnstreamer_customfilter_opencv_scaler.so"
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=640,height=480,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=\"custom\" model=\"${PATH_TO_MODEL}\" custom=\"640x480\" ! filesink location=\"testcase12.passthrough.log\" sync=true t. ! queue ! filesink location=\"testcase12.direct.log\" sync=true" 12 0 0 $PERFORMANCE
callCompareTest testcase12.direct.log testcase12.passthrough.log 12 "Compare 12" 0 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=640,height=480,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=\"custom\" model=\"${PATH_TO_MODEL}\" custom=\"320x240\" ! filesink location=\"testcase13.scaled.log\" sync=true t. ! queue ! filesink location=\"testcase13.direct.log\" sync=true" 13 0 0 $PERFORMANCE
python checkScaledTensor.py testcase13.direct.log 640 480 testcase13.scaled.log 320 240 3
testResult $? 13 "Golden test comparison" 0 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=640,height=480,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=\"custom\" model=\"${PATH_TO_MODEL}\" custom=\"1920x1080\" ! filesink location=\"testcase14.scaled.log\" sync=true t. ! queue ! filesink location=\"testcase14.direct.log\" sync=true" 14 0 0 $PERFORMANCE
python checkScaledTensor.py testcase14.direct.log 640 480 testcase14.scaled.log 1920 1080 3
testResult $? 14 "Golden test comparison" 0 1

# Test average using OpenCV (15)
# custom version
PATH_TO_MODEL_A="../../build/nnstreamer_example/custom_example_average/libnnstreamer_customfilter_average.so"
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=640,height=480,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_filter framework=\"custom\" model=\"${PATH_TO_MODEL_A}\" ! filesink location=\"testcase15.average.log\" sync=true" 15 0 0 $PERFORMANCE

# OpenCV version
PATH_TO_MODEL_A="../../build/nnstreamer_example/custom_example_opencv/libnnstreamer_customfilter_opencv_average.so"
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=640,height=480,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_filter framework=\"custom\" model=\"${PATH_TO_MODEL_A}\" ! filesink location=\"testcase15.opencv.average.log\" sync=true" 15 0 0 $PERFORMANCE

callCompareTest testcase15.opencv.average.log testcase15.average.log 15 "Compare 15" 0 0

report
