#!/usr/bin/env bash
source ../testAPI.sh

if [ "$SKIPGEN" == "YES" ]
then
  echo "Test Case Generation Skipped"
  sopath=$2
else
  echo "Test Case Generation Started"
  python ../nnstreamer_converter/generateGoldenTestResult.py 8
  sopath=$1
fi

# Test gst availability. (0)
gstTest "videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=280,height=40,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! filesink location=\"testcase.apitest.log\" sync=true" 0

# Test with small stream (1, 2)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=dimchg option=0:2 ! filesink location=\"testcase01.dimchg02.log\" sync=true t. ! queue ! filesink location=\"testcase01.direct.log\" sync=true" 1
# dim1 = 3 (RGB)
# rbs = 16 x 16 x 3
# es = 1
python checkResult.py dimchg0:b testcase01.direct.log testcase01.dimchg02.log 3 768 1
casereport 1 $? "Golden test comparison"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=BGRx ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=dimchg option=0:2 ! filesink location=\"testcase02.dimchg02.log\" sync=true t. ! queue ! filesink location=\"testcase02.direct.log\" sync=true" 2
# dim1 = 3 (RGB)
# rbs = 16 x 16 x 3
# es = 1
python checkResult.py dimchg0:b testcase02.direct.log testcase02.dimchg02.log 4 1024 1
casereport 2 $? "Golden test comparison"

report
