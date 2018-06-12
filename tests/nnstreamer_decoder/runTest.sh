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

function do_test {
    param="${4}_gen_golden_data"
    gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=testcase01_${1}_${2}x${3}.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=${1},width=${2},height=${3},framerate=0/1 ! filesink location=\"testcase01_${1}_${2}x${3}.golden.raw\" sync=true" $param

    gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=testcase01_${1}_${2}x${3}.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=${1},width=${2},height=${3},framerate=0/1 ! tensor_converter ! tensordec ! filesink location=\"testcase01_${1}_${2}x${3}.log\" sync=true" ${4}

    compareAllSizeLimit testcase01_${1}_${2}x${3}.log testcase01_${1}_${2}x${3}.golden.raw ${4}
}

do_test RGB 640 480 1

report

