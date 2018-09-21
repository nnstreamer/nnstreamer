#!/usr/bin/env bash
source ../testAPI.sh

if [ "$SKIPGEN" == "YES" ]
then
  echo "Test Case Generation Skipped"
  sopath=$2
else
  echo "Test Case Generation Started"
  python generateGoldenTestResult.py
  python ../nnstreamer_converter/generateGoldenTestResult.py 8
  sopath=$1
fi
convertBMP2PNG

function do_test {
    param="${4}_gen_golden_data"
    gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=testcase01_${1}_${2}x${3}.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=${1},width=${2},height=${3},framerate=0/1 ! filesink location=\"testcase01_${1}_${2}x${3}.golden.raw\" sync=true" $param

    gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=testcase01_${1}_${2}x${3}.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=${1},width=${2},height=${3},framerate=0/1 ! tensor_converter ! tensor_decoder ! filesink location=\"testcase01_${1}_${2}x${3}.log\" sync=true" ${4}

    compareAllSizeLimit testcase01_${1}_${2}x${3}.log testcase01_${1}_${2}x${3}.golden.raw ${4}
}

do_test RGB 640 480 1
do_test RGB 642 480 1-1
do_test BGRx 640 480 1-2
do_test BGRx 642 480 1-3

# Test with a stream of 10 small PNG frames
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_decoder ! filesink location=\"testcase2.dec.log\" sync=true t. ! queue ! filesink location=\"testcase2.con.log\" sync=true" 2
compareAll testcase2.con.log testcase2.dec.log 2

report

