#!/usr/bin/env bash
source ../testAPI.sh

if [ "$SKIPGEN" == "YES" ]
then
  echo "Test Case Generation Skipped"
  sopath=$2
else
  echo "Test Case Generation Started"
  python generateGoldenTestResult.py
  sopath=$1
fi

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=280,height=40,framerate=0/1 ! tee name=t ! queue ! videoconvert ! video/x-raw, format=BGRx ! tensor_converter silent=TRUE ! filesink location=\"test.bgrx.log\" sync=true t. ! queue ! filesink location=\"test.rgb.log\" sync=true" 1

compareAllSizeLimit testcase01.bgrx.golden test.bgrx.log 1-1
compareAllSizeLimit testcase01.rgb.golden test.rgb.log 1-2

function do_test {
	gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=testcase02_${1}_${2}x${3}.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=${1},width=${2},height=${3},framerate=0/1 ! tensor_converter silent=TRUE ! filesink location=\"testcase02_${1}_${2}x${3}.log\" sync=true" ${4}

	compareAllSizeLimit testcase02_${1}_${2}x${3}.golden testcase02_${1}_${2}x${3}.log ${4}
}
function do_test_nonip {
	gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=testcase02_${1}_${2}x${3}.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=${1},width=${2},height=${3},framerate=0/1 ! tensor_converter silent=TRUE force_memcpy=TRUE ! filesink location=\"testcase02_${1}_${2}x${3}.nonip.log\" sync=true" ${4}

	compareAllSizeLimit testcase02_${1}_${2}x${3}.golden testcase02_${1}_${2}x${3}.nonip.log ${4}
}

do_test BGRx 640 480 2-1
do_test RGB 640 480 2-2
do_test BGRx 642 480 2-3
do_test RGB 642 480 2-4

do_test_nonip BGRx 640 480 3-1
do_test_nonip RGB 640 480 3-2
do_test_nonip BGRx 642 480 3-3
do_test_nonip RGB 642 480 3-4

# @TODO Change this when YUV becomes supported by tensor_converter
# Fail Test: YUV is given
gstFailTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=YUV,width=280,height=40,framerate=0/1 ! videoconvert ! video/x-raw, format=YUV ! tensor_converter silent=TRUE ! filesink location=\"test.yuv.fail.log\" sync=true" 5-F

# Fail Test: Unknown property is given
gstFailTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=280,height=40,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter silent=TRUE whatthehell=isthis ! filesink location=\"test.yuv.fail.log\" sync=true" 6-F

# @TODO Change this whey audio stream is supported
# Fail Test: Audio ig given
gstFailTest "--gst-plugin-path=${PATH_TO_PLUGIN} audiotestsrc ! audio/x-raw ! tensor_converter silent=TRUE ! filesink location=\"test..audio.fail.log\" sync=true" 7-F

# Stream test case (genCase08 in generateGoldenTestResult.py)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! tensor_converter ! filesink location=\"testcase08.log\"" 8
compareAll testcase08.golden testcase08.log 8

report
