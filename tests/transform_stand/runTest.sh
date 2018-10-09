#!/usr/bin/env bash
source ../testAPI.sh

if [ "$SKIPGEN" == "YES" ]
then
  echo "Test Case Generation Skipped"
  sopath=$2
else
  echo "Test Case Generation Started"
  python generateTest.py
  sopath=$1
fi

# Test gst availability. (0)
gstTest "videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=280,height=40,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! filesink location=\"testcase.apitest.log\" sync=true" 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"test_%02d.dat\" caps=\"application/octet-stream\" ! tensor_converter input-dim=50:100:1:1 input-type=float32 ! tensor_transform mode=stand option=default ! multifilesink location=\"./result_%02d.log\" sync=true" 1

python checkResult.py standardization test_00.dat.golden result_00.log 4 4 f f default

casereport 1 $? "Golden test comparison"

report
