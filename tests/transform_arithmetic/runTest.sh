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
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=typecast option=float32 ! tee name=t ! queue ! tensor_transform mode=arithmetic option=add:-10 ! filesink location=\"testcase01.arithmetic.log\" sync=true t. ! queue ! filesink location=\"testcase01.direct.log\" sync=true" 1

python checkResult.py arithmetic testcase01.direct.log testcase01.arithmetic.log 4 4 f f add -10 0
casereport 1 $? "Golden test comparison"

# Test with small stream (1, 2)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=typecast option=float32 ! tee name=t ! queue ! tensor_transform mode=arithmetic option=mul:2.0 ! filesink location=\"testcase02.arithmetic.log\" sync=true t. ! queue ! filesink location=\"testcase02.direct.log\" sync=true" 2

python checkResult.py arithmetic testcase02.direct.log testcase02.arithmetic.log 4 4 f f mul 2.0 0
casereport 2 $? "Golden test comparison"

# Test for mul with floating-point operand
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=typecast option=float32 ! tee name=t ! queue ! tensor_transform mode=arithmetic option=mul:-5.5 ! filesink location=\"testcase03.arithmetic.log\" sync=true t. ! queue ! filesink location=\"testcase03.direct.log\" sync=true" 3

python checkResult.py arithmetic testcase03.direct.log testcase03.arithmetic.log 4 4 f f mul -5.5 0
casereport 3 $? "Golden test comparison"

# Fail Test 3-F: for mul with floating-point operand
gstFailTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=typecast option=float32 ! tee name=t ! queue ! tensor_transform mode=arithmetic option=mul::-5.5 ! filesink location=\"testcase03.arithmetic.fail.log\" sync=true t. ! queue ! filesink location=\"testcase03.direct.fail.log\" sync=true" 3-F

python checkResult.py arithmetic testcase03.direct.fail.log testcase03.arithmetic.fail.log 4 4 f f mul 0 -5.5
casereport 3-F $? "Golden test comparison"

# Test for add with floating-point operand
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=typecast option=float64 ! tee name=t ! queue ! tensor_transform mode=arithmetic option=add:9.900000e-001 ! filesink location=\"testcase04.arithmetic.log\" sync=true t. ! queue ! filesink location=\"testcase04.direct.log\" sync=true" 4

python checkResult.py arithmetic testcase04.direct.log testcase04.arithmetic.log 8 8 d d add 9.900000e-001 0
casereport 4 $? "Golden test comparison"

# Test for add with two floating-point operand: the second operand will be ignored.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=typecast option=float64 ! tee name=t ! queue ! tensor_transform mode=arithmetic option=add:9.900000e-001:-80.256 ! filesink location=\"testcase04.arithmetic.ok.log\" sync=true t. ! queue ! filesink location=\"testcase04.direct.ok.log\" sync=true" 4-OK

python checkResult.py arithmetic testcase04.direct.ok.log testcase04.arithmetic.ok.log 8 8 d d add 9.900000e-001 -80.256
casereport 4-OK $? "Golden test comparison"

# Test for add-mul with floating-point operands
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=typecast option=float64 ! tee name=t ! queue ! tensor_transform mode=arithmetic option=add-mul:-9.3:-11.4823e-002 ! filesink location=\"testcase05.arithmetic.log\" sync=true t. ! queue ! filesink location=\"testcase05.direct.log\" sync=true" 5

casereport 5 $? "Golden test comparison"
python checkResult.py arithmetic testcase05.direct.log testcase05.arithmetic.log 8 8 d d add-mul -9.3 -11.4823e-002

# Fail Test 5-F1: add-mul with single floating-point operand
gstFailTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=typecast option=float64 ! tee name=t ! queue ! tensor_transform mode=arithmetic option=add-mul:-9.3 ! filesink location=\"testcase05.arithmetic.fail1.log\" sync=true t. ! queue ! filesink location=\"testcase05.direct.fail1.log\" sync=true" 5-F1

casereport 5-F1 $? "Golden test comparison"
python checkResult.py arithmetic testcase05.direct.fail1.log testcase05.arithmetic.fail1.log 8 8 d d add-mul -9.3 0

# Fail Test 5-F2: add-mul with single floating-point operand
gstFailTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=typecast option=float64 ! tee name=t ! queue ! tensor_transform mode=arithmetic option=add-mul:-9.3: ! filesink location=\"testcase05.arithmetic.fail2.log\" sync=true t. ! queue ! filesink location=\"testcase05.direct.fail2.log\" sync=true" 5-F2

casereport 5-F2 $? "Golden test comparison"
python checkResult.py arithmetic testcase05.direct.fail2.log testcase05.arithmetic.fail2.log 8 8 d d add-mul -9.3 0

# Fail Test 5-F3: add-mul with single floating-point operand
gstFailTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=typecast option=float64 ! tee name=t ! queue ! tensor_transform mode=arithmetic option=add-mul::-11.4823e-002 ! filesink location=\"testcase05.arithmetic.fail3.log\" sync=true t. ! queue ! filesink location=\"testcase05.direct.fail3.log\" sync=true" 5-F3

casereport 5-F3 $? "Golden test comparison"
python checkResult.py arithmetic testcase05.direct.fail3.log testcase05.arithmetic.fail3.log 8 8 d d add-mul 30 -11.4823e-002

# Test for mul-add with floating-point operands
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tensor_transform mode=typecast option=float64 ! tee name=t ! queue ! tensor_transform mode=arithmetic option=mul-add:-50.0987e+003:15.3 ! filesink location=\"testcase06.arithmetic.log\" sync=true t. ! queue ! filesink location=\"testcase06.direct.log\" sync=true" 6

casereport 6 $? "Golden test comparison"
python checkResult.py arithmetic testcase06.direct.log testcase06.arithmetic.log 8 8 d d add-mul -50.0987e+003 15.3

report
