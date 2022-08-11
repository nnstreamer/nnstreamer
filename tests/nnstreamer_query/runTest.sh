#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-2.1-only
#
# @file runTest.sh
# @author Gichan Jang <gichan2.jang@samsung.com>
# @date Aug 25 2021
# @brief SSAT Test Cases for tensor query
#
if [[ "$SSATAPILOADED" != "1" ]]; then
    SILENT=0
    INDEPENDENT=1
    search="ssat-api.sh"
    source $search
    printf "${Blue}Independent Mode${NC}"
fi

# This is compatible with SSAT (https://github.com/myungjoo/SSAT)
testInit $1

PATH_TO_PLUGIN="../../build"
PERFORMANCE=0
# The client has to wait enough time for the server to be ready.
# Related issue: https://github.com/nnstreamer/nnstreamer/issues/3657
SLEEPTIME_SEC=5
TIMEOUT_SEC=10

check_query=$(gst-inspect-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc | grep TensorQueryServerSrc)
if [[ ! $check_query ]]; then
    echo "Cannot find tensor query plugins. Skip tests."
    report
    exit
fi

## @brief Execute file comparison test if the file exist
function _callCompareTest() {
    if [[ ! -f "$1" || ! -f "$2" ]]; then
        echo "$1 or $2 don't exist."
        return
    fi

    callCompareTest $1 $2 $3 "$4" $5 $6
}

# Run tensor query server as echo server with default address option.
PORT=`python3 ../get_available_port.py`
gstTestBackground "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc port=${PORT} ! other/tensors,format=static,num_tensors=1,dimensions=(string)3:300:300:1,types=(string)uint8 ! tensor_query_serversink async=false" 1-1 0 0 30
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc is-live=true num-buffers=10 ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! tee name = t t. ! queue ! multifilesink location= raw1_%1d.log t. ! queue ! tensor_query_client dest-port=${PORT} ! multifilesink location=result1_%1d.log sync=true " 1-2 0 0 $PERFORMANCE
_callCompareTest raw1_0.log result1_0.log 1-3 "Compare 1-3" 1 0
_callCompareTest raw1_1.log result1_1.log 1-4 "Compare 1-4" 1 0
_callCompareTest raw1_2.log result1_2.log 1-5 "Compare 1-5" 1 0
# Since the server operates in the background, wait for the server to stop before starting the next test.
kill -9 $pid &> /dev/null
wait $pid

# Run tensor query server as echo server with given address option. (multi clients)
PORT1=`python3 ../get_available_port.py`
gstTestBackground "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc host=127.0.0.1 port=${PORT1} ! other/tensors,format=static,num_tensors=1,dimensions=(string)3:300:300:1,types=(string)uint8 ! tensor_query_serversink async=false" 2-1 0 0 30
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc is-live=true num-buffers=10  ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! tee name = t t. ! queue ! multifilesink location= raw2_%1d.log t. ! queue ! tensor_query_client host=127.0.0.1 port=0 dest-host=127.0.0.1 dest-port=${PORT1} ! multifilesink location=result2_%1d.log" 2-2 0 0 $PERFORMANCE
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc is-live=true num-buffers=10  ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! tee name = t t. ! queue ! multifilesink location= raw2_2_%1d.log t. ! queue ! tensor_query_client host=127.0.0.1 port=0 dest-host=127.0.0.1 dest-port=${PORT1} ! multifilesink location=result2_2_%1d.log" 2-3 0 0 $PERFORMANCE
_callCompareTest raw2_0.log result2_0.log 2-4 "Compare 2-4" 1 0
_callCompareTest raw2_1.log result2_1.log 2-5 "Compare 2-5" 1 0
_callCompareTest raw2_2.log result2_2.log 2-6 "Compare 2-6" 1 0
_callCompareTest raw2_2_0.log result2_2_0.log 2-7 "Compare 2-7" 1 0
_callCompareTest raw2_2_1.log result2_2_1.log 2-8 "Compare 2-8" 1 0
_callCompareTest raw2_2_2.log result2_2_2.log 2-9 "Compare 2-9" 1 0
kill -9 $pid &> /dev/null
wait $pid

# Test flexible tensors
PORT=`python3 ../get_available_port.py`
gstTestBackground "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc port=${PORT} ! other/tensors,format=flexible ! tensor_query_serversink async=false" 3-1 0 0 30
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc is-live=true num-buffers=10  ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! other/tensors,format=flexible ! tee name = t t. ! queue ! multifilesink location= raw3_%1d.log t. ! queue ! tensor_query_client dest-port=${PORT} ! multifilesink location=result3_%1d.log" 3-2 0 0 $PERFORMANCE
_callCompareTest raw3_0.log result3_0.log 3-3 "Compare 3-3" 1 0
_callCompareTest raw3_1.log result3_1.log 3-4 "Compare 3-4" 1 0
_callCompareTest raw3_2.log result3_2.log 3-5 "Compare 3-5" 1 0
kill -9 $pid &> /dev/null
wait $pid

# Test multiple query server src and sink.
PORT1=`python3 ../get_available_port.py`
PORT2=`python3 ../get_available_port.py`
gstTestBackground "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc id=0 port=${PORT1} ! other/tensors,format=flexible ! tensor_query_serversink id=0 async=false \
    tensor_query_serversrc id=1 port=${PORT2} ! other/tensors,format=flexible ! tensor_query_serversink id=1 async=false" 5-1 0 0 30
# Client pipeline 5-2 is connected to server ID 0.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    videotestsrc is-live=true num-buffers=10  ! videoconvert ! videoscale ! video/x-raw,width=640,height=480,format=RGB ! \
    tensor_converter ! other/tensors,format=flexible ! tee name=t \
        t. ! queue ! multifilesink location= raw5_2_%1d.log \
        t. ! queue ! tensor_query_client dest-port=${PORT1} ! multifilesink location=result5_2_%1d.log" 5-2 0 0 $PERFORMANCE
# Client pipeline 5-3 is connected to server ID 1.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    videotestsrc is-live=true pattern=13 num-buffers=10  ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! \
    tensor_converter ! other/tensors,format=flexible ! tee name=t \
        t. ! queue ! multifilesink location= raw5_3_%1d.log \
        t. ! queue ! tensor_query_client dest-port=${PORT2} ! multifilesink location=result5_3_%1d.log" 5-3 0 0 $PERFORMANCE
_callCompareTest raw5_2_0.log result5_2_0.log 5-4 "Compare 5-4" 1 0
_callCompareTest raw5_2_1.log result5_2_1.log 5-5 "Compare 5-5" 1 0
_callCompareTest raw5_2_2.log result5_2_2.log 5-6 "Compare 5-6" 1 0
_callCompareTest raw5_3_0.log result5_3_0.log 5-7 "Compare 5-7" 1 0
_callCompareTest raw5_3_1.log result5_3_1.log 5-8 "Compare 5-8" 1 0
_callCompareTest raw5_3_2.log result5_3_2.log 5-9 "Compare 5-9" 1 0
kill -9 $pid &> /dev/null
wait $pid

# Sever src cap: Video, Server sink cap: Viedo test
PORT=`python3 ../get_available_port.py`
gstTestBackground "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc port=${PORT} ! video/x-raw,width=300,height=300,format=RGB,framerate=0/1 ! tensor_query_serversink async=false" 6-1 0 0 30
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc is-live=true num-buffers=10  ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tee name = t t. ! queue ! multifilesink location= raw6_%1d.log t. ! queue ! tensor_query_client dest-port=${PORT} ! multifilesink location=result6_%1d.log" 6-2 0 0 $PERFORMANCE
_callCompareTest raw6_0.log result6_0.log 6-3 "Compare 6-3" 1 0
_callCompareTest raw6_1.log result6_1.log 6-4 "Compare 6-4" 1 0
_callCompareTest raw6_2.log result6_2.log 6-5 "Compare 6-5" 1 0
kill -9 $pid &> /dev/null
wait $pid

# Sever src cap: Video, Server sink cap: Tensor test
PORT=`python3 ../get_available_port.py`
gstTestBackground "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc port=${PORT} ! video/x-raw,width=300,height=300,format=RGB,framerate=0/1 ! tensor_converter ! tensor_query_serversink async=false" 7-1 0 0 30
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc is-live=true num-buffers=10  ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tee name = t t. ! queue ! multifilesink location= raw7_%1d.log t. ! queue ! tensor_query_client dest-port=${PORT} ! multifilesink location=result7_%1d.log" 7-2 0 0 $PERFORMANCE
_callCompareTest raw7_0.log result7_0.log 7-3 "Compare 7-3" 1 0
_callCompareTest raw7_1.log result7_1.log 7-4 "Compare 7-4" 1 0
_callCompareTest raw7_2.log result7_2.log 7-5 "Compare 7-5" 1 0
kill -9 $pid &> /dev/null
wait $pid

# Sever src cap: Tensor, Server sink cap: Video test
PORT=`python3 ../get_available_port.py`
gstTestBackground "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc port=${PORT} ! other/tensors,format=static,num_tensors=1,dimensions=(string)3:300:300:1,types=(string)uint8,framerate=0/1 ! tensor_decoder mode=direct_video ! videoconvert ! tensor_query_serversink async=false" 8-1 0 0 30
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc is-live=true num-buffers=10  ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! tee name = t t. ! queue ! multifilesink location= raw8_%1d.log t. ! queue ! tensor_query_client dest-port=${PORT} ! multifilesink location=result8_%1d.log" 8-2 0 0 $PERFORMANCE
_callCompareTest raw8_0.log result8_0.log 8-3 "Compare 8-3" 1 0
_callCompareTest raw8_1.log result8_1.log 8-4 "Compare 8-4" 1 0
_callCompareTest raw8_2.log result8_2.log 8-5 "Compare 8-5" 1 0
kill -9 $pid &> /dev/null
wait $pid

# Check whether mqtt broker is running or not
pid=`ps aux | grep mosquitto | grep -v grep | awk '{print $2}'`
if [ $pid > 0 ]; then
    echo "mosquitto broker is running"
else
    echo "Warning! mosquitto is not running so skip Query-hybrid test."
    rm *.log
    report
    exit
fi

# Test Query-hybrid. Get server info from broker.
gstTestBackground "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc id=12345 port=0 topic=passthrough connect-type=HYBRID ! other/tensors,format=flexible,framerate=0/1 ! tee name = t t. ! queue ! multifilesink location=server1_%1d.log t. ! queue ! tensor_query_serversink id=12345 connect-type=HYBRID async=false" 4-1 0 0 5
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=50 is-live=true ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! other/tensors,format=flexible ! tee name = t t. ! queue ! multifilesink location= raw4_%1d.log t. ! queue ! tensor_query_client connect-type=HYBRID dest-host=tcp://localhost dest-port=1883 topic=passthrough ! multifilesink location=result4_%1d.log" 4-3 0 0 $PERFORMANCE
_callCompareTest raw4_0.log result4_0.log 4-4 "Compare the raw file and client received file 4-4" 1 0
_callCompareTest raw4_1.log result4_1.log 4-5 "Compare the raw file and client received file 4-5" 1 0
_callCompareTest raw4_2.log result4_2.log 4-6 "Compare the raw file and client received file 4-6" 1 0
_callCompareTest server1_0.log result4_0.log 4-7 "Compare the server 1 received file and client received file 4-7" 1 0
_callCompareTest server1_1.log result4_1.log 4-8 "Compare the server 1 received file and client received file 4-8" 1 0
_callCompareTest server1_2.log result4_2.log 4-9 "Compare the server 1 received file and client received file 4-9" 1 0

kill -9 $pid &> /dev/null
wait $pid

rm *.log

report
