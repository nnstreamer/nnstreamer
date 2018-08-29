#!/usr/bin/env bash
source ../testAPI.sh

if [ "$SKIPGEN" == "YES" ]
then
  echo "Test Case Generation Skipped"
  sopath=$2
else
  echo "Test Case Generation Started"
  python ../nnstreamer_converter/generateGoldenTestResult.py 11
  sopath=$1
fi
convertBMP2PNG

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} --gst-debug=tensordsplit:5 filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw, format = RGB, width=100, height=100, framerate=0/1 ! tensor_converter ! tensorsplit name=split tensorseg=1:100:100:3 split. ! queue ! filesink location=split00.log" 1

compareAllSizeLimit testcase_0_0.golden split00.log 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} --gst-debug=tensordsplit:5 filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw, format = RGB, width=100, height=100, framerate=0/1 ! tensor_converter ! tensorsplit name=split tensorseg=1:100:100:1,1:100:100:2 split. ! queue ! filesink location=split01_0.log split. ! queue ! filesink location=split01_1.log" 2

compareAllSizeLimit testcase_1_0.golden split01_0.log 2_0
compareAllSizeLimit testcase_1_1.golden split01_1.log 2_1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} --gst-debug=tensordsplit:5 filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw, format = RGB, width=100, height=100, framerate=0/1 ! tensor_converter ! tensorsplit name=split tensorseg=1:100:100:1,1:100:100:1,1:100:100:1 split. ! queue ! filesink location=split02_0.log split. ! queue ! filesink location=split02_1.log split. ! queue ! filesink location=split02_2.log" 3

compareAllSizeLimit testcase_2_0.golden split02_0.log 3_0
compareAllSizeLimit testcase_2_1.golden split02_1.log 3_1
compareAllSizeLimit testcase_2_2.golden split02_2.log 3_2

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} --gst-debug=tensordsplit:5 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! tensorsplit name=split tensorseg=1:16:16:3 split. ! queue ! filesink location=split03.log" 4

compareAllSizeLimit testcase_stream.golden split03.log 4

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} --gst-debug=tensordsplit:5 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! tensorsplit name=split tensorseg=1:16:16:1,1:16:16:2 split. ! queue ! filesink location=split04_0.log split. ! queue ! filesink location=split04_1.log" 5

compareAllSizeLimit testcase_stream_1_0.golden split04_0.log 5_0
compareAllSizeLimit testcase_stream_1_1.golden split04_1.log 5_1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} --gst-debug=tensordsplit:5 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! tensorsplit name=split tensorseg=1:16:16:1,1:16:16:1,1:16:16:1 split. ! queue ! filesink location=split05_0.log split. ! queue ! filesink location=split05_1.log split. ! queue ! filesink location=split05_2.log" 6

compareAllSizeLimit testcase_stream_2_0.golden split05_0.log 6_0
compareAllSizeLimit testcase_stream_2_1.golden split05_1.log 6_1
compareAllSizeLimit testcase_stream_2_2.golden split05_2.log 6_2

report
