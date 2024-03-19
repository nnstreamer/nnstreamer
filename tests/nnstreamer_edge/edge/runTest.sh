#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author Yechan Choi <yechan9.choi@samsung.com>
## @date 20 Jul 2022
## @brief SSAT Test Cases for NNStreamer
##
if [[ "$SSATAPILOADED" != "1" ]]; then
    SILENT=0
    INDEPENDENT=1
    search="ssat-api.sh"
    source $search
    printf "${Blue}Independent Mode${NC}"
fi

# This is compatible with SSAT (https://github.com/myungjoo/SSAT)
testInit $1

# NNStreamer and plugins path for test
PATH_TO_PLUGIN="../../build"
PERFORMANCE=0

check_edgesink=$(gst-inspect-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} edgesink | grep EdgeSink)
if [[ ! $check_edgesink ]]; then
    echo 'Cannot find edge sink plugins. Skip tests.'
    report
    exit
fi

## @brief Execute file comparison test if both files exist
function callCompareTestIfExist() {
    if [[ ! -f "$1" ]]; then
        echo "$1 don't exist. Skip $3($4)."
        return
    fi

    if [[ ! -f "$2" ]]; then
        echo "$2 don't exist. Skip $3($4)."
        return
    fi

    callCompareTest $1 $2 $3 "$4" $5 $6
}

## @brief Find the number of the first file that matches
## @param $1 Raw file prefix
## @param $2 Raw file suffix
## @param $3 Result file name
## @param $4 Test Case ID
function findFirstMatchedFileNumber() {
    num=-1
    result_file="$3"

    if [[ ! -f "$result_file" ]]; then
        echo "$3 don't exist."
        return
    fi

    command -v cmp
    
    if [[ $? == 0 && ! -L $(which cmp) ]]; then
        cmp_exist=1
    else
        cmp_exist=0
    fi

    while :
    do  
        num=$((num+1))
        raw_file="$1$num$2"

        if [[ ! -f "$raw_file" ]]; then
            echo "$raw_file don't exist."
            num=-1
            return
        fi

        if [[ "$cmp_exist" -eq "1" ]]; then
            # use cmp - callCompareTest is too verbose
            cmp $raw_file $result_file
            output=$?
            if [[ "$output" -eq "1" ]]; then
                output=0
            else
                output=1
            fi
        else
            # use `callCompareTest` if cmp is not exist
            callCompareTest $raw_file $result_file "$4-$num" "find the number of the first file that matches for test $4" 0 1
        fi

        if [[ "$output" -eq "1" ]]; then
            return
        fi
    done
}

# Run edge sink server as echo server with default address option. 
PORT=`python3 ../../get_available_port.py`
gstTestBackground "--gst-plugin-path=${PATH_TO_PLUGIN} \
    videotestsrc is-live=true ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tee name=t \
        t. ! queue ! multifilesink location=raw1_%1d.log \
        t. ! queue ! edgesink port=${PORT} async=false" 1-1 0 0 30
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    edgesrc dest-port=${PORT} num-buffers=10 ! multifilesink location=result1_%1d.log" 1-2 0 0 $PERFORMANCE
findFirstMatchedFileNumber "raw1_" ".log" "result1_0.log" 1-3
callCompareTestIfExist raw1_$((num+0)).log result1_0.log 1-3 "Compare 1-3" 1 0
callCompareTestIfExist raw1_$((num+1)).log result1_1.log 1-4 "Compare 1-4" 1 0
callCompareTestIfExist raw1_$((num+2)).log result1_2.log 1-5 "Compare 1-5" 1 0
callCompareTestIfExist raw1_$((num+3)).log result1_3.log 1-6 "Compare 1-6" 1 0
callCompareTestIfExist raw1_$((num+4)).log result1_4.log 1-7 "Compare 1-7" 1 0
callCompareTestIfExist raw1_$((num+5)).log result1_5.log 1-8 "Compare 1-8" 1 0
callCompareTestIfExist raw1_$((num+6)).log result1_6.log 1-9 "Compare 1-9" 1 0
callCompareTestIfExist raw1_$((num+7)).log result1_7.log 1-10 "Compare 1-10" 1 0
callCompareTestIfExist raw1_$((num+8)).log result1_8.log 1-11 "Compare 1-11" 1 0
callCompareTestIfExist raw1_$((num+9)).log result1_9.log 1-12 "Compare 1-12" 1 0
kill -9 $pid &> /dev/null
wait $pid

# Run edge sink server as echo server with default address option. (multi clients)
PORT=`python3 ../../get_available_port.py`
gstTestBackground "--gst-plugin-path=${PATH_TO_PLUGIN} \
    videotestsrc is-live=true ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tee name=t \
        t. ! queue ! multifilesink location=raw2_%1d.log \
        t. ! queue ! edgesink port=${PORT} async=false" 2-1 0 0 30
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    edgesrc dest-port=${PORT} num-buffers=10 ! multifilesink location=result2_0_%1d.log" 2-2 0 0 $PERFORMANCE
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    edgesrc dest-port=${PORT} num-buffers=10 ! multifilesink location=result2_1_%1d.log" 2-3 0 0 $PERFORMANCE
findFirstMatchedFileNumber "raw2_" ".log" "result2_0_0.log" 2-4
callCompareTestIfExist raw2_$((num+0)).log result2_0_0.log 2-4 "Compare 2-4" 1 0
callCompareTestIfExist raw2_$((num+1)).log result2_0_1.log 2-5 "Compare 2-5" 1 0
callCompareTestIfExist raw2_$((num+2)).log result2_0_2.log 2-6 "Compare 2-6" 1 0
findFirstMatchedFileNumber "raw2_" ".log" "result2_1_0.log" 2-7
callCompareTestIfExist raw2_$((num+0)).log result2_1_0.log 2-7 "Compare 2-7" 1 0
callCompareTestIfExist raw2_$((num+1)).log result2_1_1.log 2-8 "Compare 2-8" 1 0
callCompareTestIfExist raw2_$((num+2)).log result2_1_2.log 2-9 "Compare 2-9" 1 0
kill -9 $pid &> /dev/null
wait $pid

# Cap: Flexible Tensor
PORT=`python3 ../../get_available_port.py`
gstTestBackground "--gst-plugin-path=${PATH_TO_PLUGIN} \
    videotestsrc is-live=true ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! other/tensors,format=flexible ! tee name=t \
        t. ! queue ! multifilesink location=raw3_%1d.log \
        t. ! queue ! edgesink port=${PORT} async=false" 3-1 0 0 30
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    edgesrc dest-port=${PORT} num-buffers=10 ! multifilesink location=result3_%1d.log" 3-2 0 0 $PERFORMANCE
findFirstMatchedFileNumber "raw3_" ".log" "result3_0.log" 3-3
callCompareTestIfExist raw3_$((num+0)).log result3_0.log 3-3 "Compare 3-3" 1 0
callCompareTestIfExist raw3_$((num+1)).log result3_1.log 3-4 "Compare 3-4" 1 0
callCompareTestIfExist raw3_$((num+2)).log result3_2.log 3-5 "Compare 3-5" 1 0
kill -9 $pid &> /dev/null
wait $pid

# Cap: Tensor
PORT=`python3 ../../get_available_port.py`
gstTestBackground "--gst-plugin-path=${PATH_TO_PLUGIN} \
    videotestsrc is-live=true ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! other/tensors,num_tensors=1,dimensions=3:300:300:1,types=uint8,format=static ! tee name=t \
        t. ! queue ! multifilesink location=raw4_%1d.log \
        t. ! queue ! edgesink port=${PORT} async=false" 4-1 0 0 30
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    edgesrc dest-port=${PORT} num-buffers=10 ! multifilesink location=result4_%1d.log" 4-2 0 0 $PERFORMANCE
findFirstMatchedFileNumber "raw4_" ".log" "result4_0.log" 4-3
callCompareTestIfExist raw4_$((num+0)).log result4_0.log 4-3 "Compare 4-3" 1 0
callCompareTestIfExist raw4_$((num+1)).log result4_1.log 4-4 "Compare 4-4" 1 0
callCompareTestIfExist raw4_$((num+2)).log result4_2.log 4-5 "Compare 4-5" 1 0
kill -9 $pid &> /dev/null
wait $pid

# Cap: Audio
PORT=`python3 ../../get_available_port.py`
gstTestBackground "--gst-plugin-path=${PATH_TO_PLUGIN} \
    audiotestsrc ! audioconvert ! tee name=t \
        t. ! queue ! multifilesink location=raw5_%1d.log \
        t. ! queue ! edgesink port=${PORT} async=false" 5-1 0 0 30
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    edgesrc dest-port=${PORT} num-buffers=10 ! multifilesink location=result5_%1d.log" 5-2 0 0 $PERFORMANCE
findFirstMatchedFileNumber "raw5_" ".log" "result5_0.log" 5-3
callCompareTestIfExist raw5_$((num+0)).log result5_0.log 5-3 "Compare 5-3" 1 0
callCompareTestIfExist raw5_$((num+1)).log result5_1.log 5-4 "Compare 5-4" 1 0
callCompareTestIfExist raw5_$((num+2)).log result5_2.log 5-5 "Compare 5-5" 1 0
kill -9 $pid &> /dev/null
wait $pid

if [ -f /usr/sbin/mosquitto ]
then
  testResult 1 9-0 "mosquitto mqtt broker search" 1
else
  testResult 0 9-0 "mosquitto mqtt broker search" 1
  rm *.log
  report
  exit
fi

# Launch mosquitto
PORT=`python3 ../../get_available_port.py`
/usr/sbin/mosquitto -p ${PORT}&
mospid=$!

gstTestBackground "--gst-plugin-path=${PATH_TO_PLUGIN} \
    videotestsrc is-live=true ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tee name=t \
        t. ! queue ! multifilesink location=raw6_%1d.log \
        t. ! queue ! edgesink port=0 connect-type=HYBRID dest-host=127.0.0.1 dest-port=${PORT} topic=tempTopic async=false" 6-1 0 0 30
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    edgesrc port=0 connect-type=HYBRID dest-host=127.0.0.1 dest-port=${PORT} topic=tempTopic num-buffers=10 ! multifilesink location=result6_0_%1d.log" 6-2 0 0 $PERFORMANCE
findFirstMatchedFileNumber "raw6_" ".log" "result6_0_0.log" 6-4
callCompareTestIfExist raw6_$((num+0)).log result6_0_0.log 6-4 "Compare 6-4" 1 0
callCompareTestIfExist raw6_$((num+1)).log result6_0_1.log 6-5 "Compare 6-5" 1 0
callCompareTestIfExist raw6_$((num+2)).log result6_0_2.log 6-6 "Compare 6-6" 1 0
kill -9 $pid &> /dev/null
wait $pid

# Check AITT lib is exist or not.
/sbin/ldconfig -p | grep libaitt-enable-later.so >/dev/null 2>&1
if [[ "$?" != 0 ]]; then
    echo "AITT-SES lib is not installed. Skip AITT test."
else
    # Data publishing via AITT
    gstTestBackground "--gst-plugin-path=${PATH_TO_PLUGIN} \
        videotestsrc is-live=true ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tee name=t \
            t. ! queue ! multifilesink location=raw7_%1d.log \
            t. ! queue ! edgesink port=0 connect-type=AITT dest-host=127.0.0.1 dest-port=${PORT} topic=tempTopic async=false" 7-1 0 0 30
    gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
        edgesrc port=0 connect-type=AITT dest-host=127.0.0.1 dest-port=${PORT} topic=tempTopic num-buffers=10 ! multifilesink location=result7_0_%1d.log" 7-2 0 0 $PERFORMANCE
    gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
        edgesrc port=0 connect-type=AITT dest-host=127.0.0.1 dest-port=${PORT} topic=tempTopic num-buffers=10 ! multifilesink location=result7_1_%1d.log" 7-3 0 0 $PERFORMANCE
    findFirstMatchedFileNumber "raw7_" ".log" "result7_0_0.log" 7-4
    callCompareTestIfExist raw7_$((num+0)).log result7_0_0.log 7-4 "Compare 7-4" 1 0
    callCompareTestIfExist raw7_$((num+1)).log result7_0_1.log 7-5 "Compare 7-5" 1 0
    callCompareTestIfExist raw7_$((num+2)).log result7_0_2.log 7-6 "Compare 7-6" 1 0
    findFirstMatchedFileNumber "raw7_" ".log" "result7_1_0.log" 7-7
    callCompareTestIfExist raw7_$((num+0)).log result7_1_0.log 7-7 "Compare 7-7" 1 0
    callCompareTestIfExist raw7_$((num+1)).log result7_1_1.log 7-8 "Compare 7-8" 1 0
    callCompareTestIfExist raw7_$((num+2)).log result7_1_2.log 7-9 "Compare 7-9" 1 0
    kill -9 $pid &> /dev/null
    wait $pid
fi

# Data publishing via MQTT
kill -9 $mospid &> /dev/null
PORT=`python3 ../../get_available_port.py`
/usr/sbin/mosquitto -p ${PORT}&
mospid=$!

gstTestBackground "--gst-plugin-path=${PATH_TO_PLUGIN} \
    videotestsrc is-live=true ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tee name=t \
        t. ! queue ! multifilesink location=raw8_%1d.log \
        t. ! queue ! edgesink port=0 connect-type=MQTT dest-host=127.0.0.1 dest-port=${PORT} topic=tempTopic async=false" 8-1 0 0 30
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    edgesrc port=0 connect-type=MQTT dest-host=127.0.0.1 dest-port=${PORT} topic=tempTopic num-buffers=10 ! multifilesink location=result8_0_%1d.log" 8-2 0 0 $PERFORMANCE
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    edgesrc port=0 connect-type=MQTT dest-host=127.0.0.1 dest-port=${PORT} topic=tempTopic num-buffers=10 ! multifilesink location=result8_1_%1d.log" 8-3 0 0 $PERFORMANCE
findFirstMatchedFileNumber "raw8_" ".log" "result8_0_0.log" 8-4
callCompareTestIfExist raw8_$((num+0)).log result8_0_0.log 8-4 "Compare 8-4" 1 0
callCompareTestIfExist raw8_$((num+1)).log result8_0_1.log 8-5 "Compare 8-5" 1 0
callCompareTestIfExist raw8_$((num+2)).log result8_0_2.log 8-6 "Compare 8-6" 1 0
findFirstMatchedFileNumber "raw8_" ".log" "result8_1_0.log" 8-7
callCompareTestIfExist raw8_$((num+0)).log result8_1_0.log 8-7 "Compare 8-7" 1 0
callCompareTestIfExist raw8_$((num+1)).log result8_1_1.log 8-8 "Compare 8-8" 1 0
callCompareTestIfExist raw8_$((num+2)).log result8_1_2.log 8-9 "Compare 8-9" 1 0
kill -9 $pid &> /dev/null
wait $pid

kill -9 $mospid &> /dev/null

rm *.log
report
exit
