#!/usr/bin/env bash
source ../testAPI.sh

if [ "$SKIPGEN" == "YES" ]
then
  echo "Test Case Generation Skipped"
  sopath=$2
else
  echo "Test Case Generation Started"
  python ../nnstreamer_converter/generateGoldenTestResult.py 9
  python generateTest.py
  sopath=$1
fi
convertBMP2PNG

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} --gst-debug=tensor_merge:5 tensor_merge name=merge mode=linear option=2 ! filesink location=testcase01_RGB_100x100.log filesrc location=testcase02_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! merge.sink_0" 1

compareAllSizeLimit testcase01_RGB_100x100.golden testcase01_RGB_100x100.log 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} --gst-debug=tensor_merge:5 tensor_merge name=merge mode=linear option=2 ! filesink location=testcase02_RGB_100x100.log filesrc location=testcase02_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! merge.sink_0 filesrc location=testcase02_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! merge.sink_1" 2

compareAllSizeLimit testcase02_RGB_100x100.golden testcase02_RGB_100x100.log 2

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} --gst-debug=tensor_merge:5 tensor_merge name=merge mode=linear option=2 ! filesink location=testcase03_RGB_100x100.log filesrc location=testcase02_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! merge.sink_0 filesrc location=testcase02_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! merge.sink_1 filesrc location=testcase02_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! merge.sink_2" 3

compareAllSizeLimit testcase03_RGB_100x100.golden testcase03_RGB_100x100.log 3

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} --gst-debug=tensor_merge:5 tensor_merge name=merge mode=linear option=2 ! filesink location=testcase01.log multifilesrc location=\"testsequence01_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_0" 4

compareAllSizeLimit testcase01.golden testcase01.log 4

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} --gst-debug=tensor_merge:5 tensor_merge name=merge mode=linear option=2 ! filesink location=testcase02.log multifilesrc location=\"testsequence02_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_0 multifilesrc location=\"testsequence02_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_1" 5

compareAllSizeLimit testcase02.golden testcase02.log 5

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} --gst-debug=tensor_merge:5 tensor_merge name=merge mode=linear option=2 ! filesink location=testcase03.log multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_0 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_1 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_2" 6

compareAllSizeLimit testcase03.golden testcase03.log 6

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} --gst-debug=tensor_merge:5 tensor_merge name=merge mode=linear option=2 ! filesink location=testcase04.log multifilesrc location=\"testsequence04_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_0 multifilesrc location=\"testsequence04_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_1 multifilesrc location=\"testsequence04_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_2 multifilesrc location=\"testsequence04_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! merge.sink_3" 7

compareAllSizeLimit testcase03.golden testcase03.log 7


gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} --gst-debug=tensor_merge:5 tensor_merge name=merge mode=linear option=0 ! filesink location=channel.log filesrc location=channel_00.dat blocksize=60000 num_buffers=1 ! application/octet-stream ! tensor_converter input-dim=3:50:100:1 input-type=float32 ! merge.sink_0 filesrc location=channel_01.dat blocksize=40000 num_buffers=1 ! application/octet-stream ! tensor_converter input-dim=2:50:100:1 input-type=float32 ! merge.sink_1 filesrc location=channel_02.dat blocksize=80000 num_buffers=1 ! application/octet-stream ! tensor_converter input-dim=4:50:100:1 input-type=float32 ! merge.sink_2" 8

compareAllSizeLimit channel.golden channel.log 8

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} --gst-debug=tensor_merge:5 tensor_merge name=merge mode=linear option=1 ! filesink location=width.log filesrc location=width_100.dat blocksize=60000 num_buffers=1 ! application/octet-stream ! tensor_converter input-dim=3:100:50:1 input-type=float32 ! merge.sink_0 filesrc location=width_200.dat blocksize=120000 num_buffers=1 ! application/octet-stream ! tensor_converter input-dim=3:200:50:1 input-type=float32 ! merge.sink_1 filesrc location=width_300.dat blocksize=180000 num_buffers=1 ! application/octet-stream ! tensor_converter input-dim=3:300:50:1 input-type=float32 ! merge.sink_2" 9

compareAllSizeLimit width.golden width.log 9

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} --gst-debug=tensor_merge:5 tensor_merge name=merge mode=linear option=3 ! filesink location=batch.log filesrc location=batch_1.dat blocksize=60000 num_buffers=1 ! application/octet-stream ! tensor_converter input-dim=3:100:50:1 input-type=float32 ! merge.sink_0 filesrc location=batch_2.dat blocksize=120000 num_buffers=1 ! application/octet-stream ! tensor_converter input-dim=3:100:50:2 input-type=float32 ! merge.sink_1 filesrc location=batch_3.dat blocksize=180000 num_buffers=1 ! application/octet-stream ! tensor_converter input-dim=3:100:50:3 input-type=float32 ! merge.sink_2" 10

compareAllSizeLimit batch.golden batch.log 10


report
