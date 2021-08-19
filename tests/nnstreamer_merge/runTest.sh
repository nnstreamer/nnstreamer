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
    python ../nnstreamer_converter/generateGoldenTestResult.py 9
    python3 generateTest.py
    sopath=$1
fi
convertBMP2PNG

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_merge name=merge mode=linear option=2 ! filesink location=testcase01_RGB_100x100.log filesrc location=testcase02_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! merge.sink_0" 1 0 0 $PERFORMANCE

callCompareTest testcase01_RGB_100x100.golden testcase01_RGB_100x100.log 1 "Compare 1" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_merge name=merge mode=linear option=2 ! filesink location=testcase02_RGB_100x100.log filesrc location=testcase02_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! merge.sink_0 filesrc location=testcase02_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! merge.sink_1" 2 0 0 $PERFORMANCE

callCompareTest testcase02_RGB_100x100.golden testcase02_RGB_100x100.log 2 "Compare 2" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_merge name=merge mode=linear option=2 ! filesink location=testcase03_RGB_100x100.log filesrc location=testcase02_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! merge.sink_0 filesrc location=testcase02_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! merge.sink_1 filesrc location=testcase02_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! merge.sink_2" 3 0 0 $PERFORMANCE

callCompareTest testcase03_RGB_100x100.golden testcase03_RGB_100x100.log 3 "Compare 3" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_merge name=merge mode=linear option=2 ! filesink location=testcase01.log multifilesrc location=\"testsequence01_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_0" 4 0 0 $PERFORMANCE

callCompareTest testcase01.golden testcase01.log 4 "Compare 4" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_merge name=merge mode=linear option=2 ! filesink location=testcase02.log multifilesrc location=\"testsequence02_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_0 multifilesrc location=\"testsequence02_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_1" 5 0 0 $PERFORMANCE

callCompareTest testcase02.golden testcase02.log 5 "Compare 5" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_merge name=merge mode=linear option=2 ! filesink location=testcase03.log multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_0 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_1 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_2" 6 0 0 $PERFORMANCE

callCompareTest testcase03.golden testcase03.log 6 "Compare 6" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_merge name=merge mode=linear option=2 ! filesink location=testcase04.log multifilesrc location=\"testsequence04_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_0 multifilesrc location=\"testsequence04_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_1 multifilesrc location=\"testsequence04_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_2 multifilesrc location=\"testsequence04_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_3" 7 0 0 $PERFORMANCE

callCompareTest testcase03.golden testcase03.log 7 "Compare 7" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_merge name=merge mode=linear option=0 ! filesink location=channel.log filesrc location=channel_00.dat blocksize=60000 num_buffers=1 ! application/octet-stream ! tensor_converter input-dim=3:50:100:1 input-type=float32 ! merge.sink_0 filesrc location=channel_01.dat blocksize=40000 num_buffers=1 ! application/octet-stream ! tensor_converter input-dim=2:50:100:1 input-type=float32 ! merge.sink_1 filesrc location=channel_02.dat blocksize=80000 num_buffers=1 ! application/octet-stream ! tensor_converter input-dim=4:50:100:1 input-type=float32 ! merge.sink_2" 8 0 0 $PERFORMANCE

callCompareTest channel.golden channel.log 8 "Compare 8" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_merge name=merge mode=linear option=1 ! filesink location=width.log filesrc location=width_100.dat blocksize=60000 num_buffers=1 ! application/octet-stream ! tensor_converter input-dim=3:100:50:1 input-type=float32 ! merge.sink_0 filesrc location=width_200.dat blocksize=120000 num_buffers=1 ! application/octet-stream ! tensor_converter input-dim=3:200:50:1 input-type=float32 ! merge.sink_1 filesrc location=width_300.dat blocksize=180000 num_buffers=1 ! application/octet-stream ! tensor_converter input-dim=3:300:50:1 input-type=float32 ! merge.sink_2" 9 0 0 $PERFORMANCE

callCompareTest width.golden width.log 9 "Compare 9" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_merge name=merge mode=linear option=3 ! filesink location=batch.log filesrc location=batch_1.dat blocksize=60000 num_buffers=1 ! application/octet-stream ! tensor_converter input-dim=3:100:50:1 input-type=float32 ! merge.sink_0 filesrc location=batch_2.dat blocksize=120000 num_buffers=1 ! application/octet-stream ! tensor_converter input-dim=3:100:50:2 input-type=float32 ! merge.sink_1 filesrc location=batch_3.dat blocksize=180000 num_buffers=1 ! application/octet-stream ! tensor_converter input-dim=3:100:50:3 input-type=float32 ! merge.sink_2" 10 0 0 $PERFORMANCE

callCompareTest batch.golden batch.log 10 "Compare 10" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_merge name=merge mode=linear option=2 silent=true sync-mode=slowest ! multifilesink location=testsynch00_%1d.log multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_0 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)10/1\" ! pngdec ! tensor_converter ! merge.sink_1" 11 0 0 $PERFORMANCE

callCompareTest testsynch00_0.golden testsynch00_0.log 11-1 "Compare 11-1" 1 0
callCompareTest testsynch00_1.golden testsynch00_1.log 11-2 "Compare 11-2" 1 0
callCompareTest testsynch00_2.golden testsynch00_2.log 11-3 "Compare 11-3" 1 0
callCompareTest testsynch00_3.golden testsynch00_3.log 11-4 "Compare 11-4" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_merge name=merge mode=linear option=2 silent=true sync-mode=slowest ! multifilesink location=testsynch01_%1d.log multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_0 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)20/1\" ! pngdec ! tensor_converter ! merge.sink_1" 12 0 0 $PERFORMANCE

callCompareTest testsynch01_0.golden testsynch01_0.log 12-1 "Compare 12-1" 1 0
callCompareTest testsynch01_1.golden testsynch01_1.log 12-2 "Compare 12-2" 1 0
callCompareTest testsynch01_2.golden testsynch01_2.log 12-3 "Compare 12-3" 1 0
callCompareTest testsynch01_3.golden testsynch01_3.log 12-4 "Compare 12-4" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_merge name=merge mode=linear option=2 silent=true sync-mode=slowest ! multifilesink location=testsynch02_%1d.log multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_0 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)20/1\" ! pngdec ! tensor_converter ! merge.sink_1 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)10/1\" ! pngdec ! tensor_converter ! merge.sink_2" 13 0 0 $PERFORMANCE

callCompareTest testsynch02_0.golden testsynch02_0.log 13-1 "Compare 13-1" 1 0
callCompareTest testsynch02_1.golden testsynch02_1.log 13-2 "Compare 13-2" 1 0
callCompareTest testsynch02_2.golden testsynch02_2.log 13-3 "Compare 13-3" 1 0
callCompareTest testsynch02_3.golden testsynch02_3.log 13-4 "Compare 13-4" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_merge name=merge mode=linear option=2 silent=true sync-mode=basepad sync-option=0:33333333 ! multifilesink location=testsynch03_%1d.log multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_0 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)10/1\" ! pngdec ! tensor_converter ! merge.sink_1" 14 0 0 $PERFORMANCE

callCompareTest testsynch03_0.golden testsynch03_0.log 14-1 "Compare 14-1" 1 0
callCompareTest testsynch03_1.golden testsynch03_1.log 14-2 "Compare 14-2" 1 0
callCompareTest testsynch03_2.golden testsynch03_2.log 14-3 "Compare 14-3" 1 0
callCompareTest testsynch03_3.golden testsynch03_3.log 14-4 "Compare 14-4" 1 0
callCompareTest testsynch03_4.golden testsynch03_4.log 14-5 "Compare 14-5" 1 0
callCompareTest testsynch03_5.golden testsynch03_5.log 14-6 "Compare 14-6" 1 0
callCompareTest testsynch03_6.golden testsynch03_6.log 14-7 "Compare 14-7" 1 0
callCompareTest testsynch03_7.golden testsynch03_7.log 14-8 "Compare 14-8" 1 0
callCompareTest testsynch03_8.golden testsynch03_8.log 14-9 "Compare 14-9" 1 0
callCompareTest testsynch03_9.golden testsynch03_9.log 14-10 "Compare 14-10" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_merge name=merge mode=linear option=2 silent=true sync-mode=basepad sync-option=0:33333333 ! multifilesink location=testsynch04_%1d.log multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_0 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)20/1\" ! pngdec ! tensor_converter ! merge.sink_1" 15 0 0 $PERFORMANCE

callCompareTest testsynch04_0.golden testsynch04_0.log 15-1 "Compare 15-1" 1 0
callCompareTest testsynch04_1.golden testsynch04_1.log 15-2 "Compare 15-2" 1 0
callCompareTest testsynch04_2.golden testsynch04_2.log 15-3 "Compare 15-3" 1 0
callCompareTest testsynch04_3.golden testsynch04_3.log 15-4 "Compare 15-4" 1 0
callCompareTest testsynch04_4.golden testsynch04_4.log 15-5 "Compare 15-5" 1 0
callCompareTest testsynch04_5.golden testsynch04_5.log 15-6 "Compare 15-6" 1 0
callCompareTest testsynch04_6.golden testsynch04_6.log 15-7 "Compare 15-7" 1 0
callCompareTest testsynch04_7.golden testsynch04_7.log 15-8 "Compare 15-8" 1 0
callCompareTest testsynch04_8.golden testsynch04_8.log 15-9 "Compare 15-9" 1 0
callCompareTest testsynch04_9.golden testsynch04_9.log 15-10 "Compare 15-10" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_merge name=merge mode=linear option=2 silent=true sync-mode=basepad sync-option=0:33333333 ! multifilesink location=testsynch05_%1d.log multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_0 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)20/1\" ! pngdec ! tensor_converter ! merge.sink_1 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)10/1\" ! pngdec ! tensor_converter ! merge.sink_2" 16 0 0 $PERFORMANCE

callCompareTest testsynch05_0.golden testsynch05_0.log 16-1 "Compare 16-1" 1 0
callCompareTest testsynch05_1.golden testsynch05_1.log 16-2 "Compare 16-2" 1 0
callCompareTest testsynch05_2.golden testsynch05_2.log 16-3 "Compare 16-3" 1 0
callCompareTest testsynch05_3.golden testsynch05_3.log 16-4 "Compare 16-4" 1 0
callCompareTest testsynch05_4.golden testsynch05_4.log 16-5 "Compare 16-5" 1 0
callCompareTest testsynch05_5.golden testsynch05_5.log 16-6 "Compare 16-6" 1 0
callCompareTest testsynch05_6.golden testsynch05_6.log 16-7 "Compare 16-7" 1 0
callCompareTest testsynch05_7.golden testsynch05_7.log 16-8 "Compare 16-8" 1 0
callCompareTest testsynch05_8.golden testsynch05_8.log 16-9 "Compare 16-9" 1 0
callCompareTest testsynch05_9.golden testsynch05_9.log 16-10 "Compare 16-10" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_merge name=merge mode=linear option=2 silent=true sync-mode=basepad sync-option=0 ! multifilesink location=testsynch06_%1d.log multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_0 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)20/1\" ! pngdec ! tensor_converter ! merge.sink_1  multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)10/1\" ! pngdec ! tensor_converter ! merge.sink_2" 17 0 0 $PERFORMANCE
callCompareTest testsynch05_0.golden testsynch06_0.log 17-1 "Compare 17-1" 1 0
callCompareTest testsynch05_1.golden testsynch06_1.log 17-2 "Compare 17-2" 1 0
callCompareTest testsynch05_2.golden testsynch06_2.log 17-3 "Compare 17-3" 1 0
callCompareTest testsynch05_3.golden testsynch06_3.log 17-4 "Compare 17-4" 1 0
callCompareTest testsynch05_4.golden testsynch06_4.log 17-5 "Compare 17-5" 1 0
callCompareTest testsynch05_5.golden testsynch06_5.log 17-6 "Compare 17-6" 1 0
callCompareTest testsynch05_6.golden testsynch06_6.log 17-7 "Compare 17-7" 1 0
callCompareTest testsynch05_7.golden testsynch06_7.log 17-8 "Compare 17-8" 1 0
callCompareTest testsynch05_8.golden testsynch06_8.log 17-9 "Compare 17-9" 1 0
callCompareTest testsynch05_9.golden testsynch06_9.log 17-10 "Compare 17-10" 1 0

# Test Case for sync-option=0 without duration. If it does not set, then it use pts(n+1) - pts(n) as base duration. If there are pts(n-1) and pts(n) avaiable within duration condition, it always take pts(n).
# For this test case, outputs are generated every 1000000000 nsec, and they are [0,0],[1000000000,133333332], [2000000000,2333333332], [3000000000,2999999997]. The reason last one is 2999999997 instead of  3333333332 is EOS of basepad.

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_merge name=merge mode=linear option=2 silent=true sync-mode=basepad sync-option=0 ! multifilesink location=testsynch07_%1d.log multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)10/1\" ! pngdec ! tensor_converter ! merge.sink_0 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_1" 18 0 0 $PERFORMANCE

callCompareTest testsynch07_0.golden testsynch07_0.log 18-1 "Compare 18-1" 1 0
callCompareTest testsynch07_1.golden testsynch07_1.log 18-2 "Compare 18-2" 1 0
callCompareTest testsynch07_2.golden testsynch07_2.log 18-3 "Compare 18-3" 1 0
callCompareTest testsynch07_3.golden testsynch07_3.log 18-4 "Compare 18-4" 1 0

# Test Case for sync-option=0:0. If the duration set 0, then it means that every tensor which is exceed base time is not going to be merged. Always it merged nearest tensor with base time.
# For this test case, outputs are generated every 1000000000 nsec, and they are [0,0],[1000000000,999999999], [2000000000,1999999999], [3000000000,2999999997].

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_merge name=merge mode=linear option=2 silent=true sync-mode=basepad sync-option=0:0 ! multifilesink location=testsynch08_%1d.log multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)10/1\" ! pngdec ! tensor_converter ! merge.sink_0 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_1" 19 0 0 $PERFORMANCE

callCompareTest testsynch08_0.golden testsynch08_0.log 19-1 "Compare 19-1" 1 0
callCompareTest testsynch08_1.golden testsynch08_1.log 19-2 "Compare 19-2" 1 0
callCompareTest testsynch08_2.golden testsynch08_2.log 19-3 "Compare 19-3" 1 0
callCompareTest testsynch08_3.golden testsynch08_3.log 19-4 "Compare 19-4" 1 0

rm *.log *.bmp *.png *.golden *.raw *.dat

report
