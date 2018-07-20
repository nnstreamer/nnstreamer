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
convertBMP2PNG

# Test gst availability. (0)
gstTest "videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=280,height=40,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! filesink location=\"testcase.apitest.log\" sync=true" 0

# Test with small stream (1, 2)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=typecast option=uint32 ! filesink location=\"testcase01.typecast.log\" sync=true t. ! queue ! filesink location=\"testcase01.direct.log\" sync=true" 1
# uint8 -> uint32
python checkResult.py typecast testcase01.direct.log testcase01.typecast.log uint8 1 B uint32 4 I
casereport 1 $? "Golden test comparison"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=BGRx ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=typecast option=uint16 ! filesink location=\"testcase02.typecast.log\" sync=true t. ! queue ! filesink location=\"testcase02.direct.log\" sync=true" 2
# uint8 -> uint32
python checkResult.py typecast testcase02.direct.log testcase02.typecast.log uint8 1 B uint16 2 H
casereport 2 $? "Golden test comparison"


gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=BGRx ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=typecast option=int8 ! filesink location=\"testcase03.typecast.log\" sync=true t. ! queue ! filesink location=\"testcase03.direct.log\" sync=true" 3
# uint8 -> int8
python checkResult.py typecast testcase03.direct.log testcase03.typecast.log uint8 1 B int8 1 b
casereport 3 $? "Golden test comparison"


gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=BGRx ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=typecast option=uint32 ! tensor_transform mode=typecast option=uint8 ! filesink location=\"testcase04.typecast.log\" sync=true t. ! queue ! filesink location=\"testcase04.direct.log\" sync=true" 4
# uint8 -> uint32 -> uint8
python checkResult.py typecast testcase04.direct.log testcase04.typecast.log uint8 1 B uint8 1 B
casereport 4 $? "Golden test comparison"

report
