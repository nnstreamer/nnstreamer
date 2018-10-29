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
convertBMP2PNG

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=280,height=40,framerate=0/1 ! tee name=t ! queue ! videoconvert ! video/x-raw, format=BGRx ! tensor_converter silent=TRUE ! filesink location=\"test.bgrx.log\" sync=true t. ! queue ! filesink location=\"test.rgb.log\" sync=true" 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=GRAY8,width=280,height=40,framerate=0/1 ! queue ! tensor_converter silent=TRUE ! filesink location=\"test.gray8.log\" sync=true" 1

compareAllSizeLimit testcase01.bgrx.golden test.bgrx.log 1-1
compareAllSizeLimit testcase01.rgb.golden test.rgb.log 1-2
compareAllSizeLimit testcase01.gray8.golden test.gray8.log 1-3

function do_test {
	gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=testcase02_${1}_${2}x${3}.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=${1},width=${2},height=${3},framerate=0/1 ! tensor_converter silent=TRUE ! filesink location=\"testcase02_${1}_${2}x${3}.log\" sync=true" ${4}

	compareAllSizeLimit testcase02_${1}_${2}x${3}.golden testcase02_${1}_${2}x${3}.log ${4}
}

do_test BGRx 640 480 2-1
do_test RGB 640 480 2-2
do_test GRAY8 640 480 2-3
do_test BGRx 642 480 2-4
do_test RGB 642 480 2-5
do_test GRAY8 642 480 2-6

# @TODO Change this when YUV becomes supported by tensor_converter
# Fail Test: YUV is given
gstFailTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=YUV,width=280,height=40,framerate=0/1 ! videoconvert ! video/x-raw, format=YUV ! tensor_converter silent=TRUE ! filesink location=\"test.yuv.fail.log\" sync=true" 5-F

# Fail Test: Unknown property is given
gstFailTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=280,height=40,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter silent=TRUE whatthehell=isthis ! filesink location=\"test.yuv.fail.log\" sync=true" 6-F

# audio format S16LE, 8k sample rate, samples per buffer 8000
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} audiotestsrc num-buffers=1 samplesperbuffer=8000 ! audioconvert ! audio/x-raw,format=S16LE,rate=8000 ! tee name=t ! queue ! audioconvert ! tensor_converter frames-per-tensor=8000 ! filesink location=\"test.audio8k.s16le.log\" sync=true t. ! queue ! filesink location=\"test.audio8k.s16le.origin.log\" sync=true" 7-1
compareAll test.audio8k.s16le.origin.log test.audio8k.s16le.log 7-2

# audio format U8, 16k sample rate, samples per buffer 8000
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} audiotestsrc num-buffers=1 samplesperbuffer=8000 ! audioconvert ! audio/x-raw,format=U8,rate=16000 ! tee name=t ! queue ! audioconvert ! tensor_converter frames-per-tensor=8000 ! filesink location=\"test.audio16k.u8.log\" sync=true t. ! queue ! filesink location=\"test.audio16k.u8.origin.log\" sync=true" 7-3
compareAll test.audio16k.u8.origin.log test.audio16k.u8.log 7-4

# audio format U16LE, 16k sample rate, 2 channels, samples per buffer 8000
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} audiotestsrc num-buffers=1 samplesperbuffer=8000 ! audioconvert ! audio/x-raw,format=U16LE,rate=16000,channels=2 ! tee name=t ! queue ! audioconvert ! tensor_converter frames-per-tensor=8000 ! filesink location=\"test.audio16k2c.u16le.log\" sync=true t. ! queue ! filesink location=\"test.audio16k2c.u16le.origin.log\" sync=true" 7-5
compareAll test.audio16k2c.u16le.origin.log test.audio16k2c.u16le.log 7-6

# audio format S32LE, 8k sample rate, samples per buffer 8000
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} audiotestsrc num-buffers=1 samplesperbuffer=8000 ! audioconvert ! audio/x-raw,format=S32LE,rate=8000 ! tensor_converter frames-per-tensor=8000 ! filesink location=\"test.audio8k.s32le.log\" sync=true" 7-7

# Stream test case (genCase08 in generateGoldenTestResult.py)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! tensor_converter ! filesink location=\"testcase08.log\"" 8
compareAll testcase08.golden testcase08.log 8

# PTS test case 
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=10 ! video/x-raw,format=GRAY8,width=280,height=40,framerate=100/1 ! tensor_converter silent=false ! filesink location=\"test.video.log\" sync=true" 9-1 &> temp
cat temp | grep pts | cut -d '=' -f 2 | cut -d ' ' -f 2 > testPTS01.log
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=10 timestamp-offset=100000000 ! video/x-raw,format=GRAY8,width=280,height=40,framerate=100/1 ! tensor_converter silent=false ! filesink location=\"test.offset.log\" sync=true" 9-2 &> temp
cat temp | grep pts | cut -d '=' -f 2 | cut -d ' ' -f 2 > testPTS02.log

compareAll testPTS01.golden testPTS01.log 9-1
compareAll testPTS02.golden testPTS02.log 9-2

report
