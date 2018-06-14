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

report
