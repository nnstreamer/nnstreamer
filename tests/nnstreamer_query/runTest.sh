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
    printf "${Blue}Independent Mode${NC}
"
fi

# This is compatible with SSAT (https://github.com/myungjoo/SSAT)
testInit $1

PATH_TO_PLUGIN="../../build"
PERFORMANCE=0
TIMEOUT_SEC=5

check_query=$(gst-inspect-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc | grep TensorQueryServerSrc)
if [[ ! $check_query ]]; then
    echo "Cannot find tensor query plugins. Skip tests."
    report
    exit
fi

# Run tensor query server as echo server with default adress option.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc num-buffers=3 ! other/tensors,num_tensors=1,dimensions=3:300:300:1,types=uint8 ! tensor_query_serversink" 1-1 0 0 $PERFORMANCE $TIMEOUT_SEC &
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! tee name = t t. ! queue ! multifilesink location= raw1_%1d.log t. ! queue ! tensor_query_client ! multifilesink location=result1_%1d.log" 1-2 0 0 $PERFORMANCE
callCompareTest raw1_0.log result1_0.log 1-3 "Compare 1-3" 1 0
callCompareTest raw1_1.log result1_1.log 1-4 "Compare 1-4" 1 0
callCompareTest raw1_2.log result1_2.log 1-5 "Compare 1-5" 1 0

# Run tensor query server as echo server with given address option. (multi clients)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc host=127.0.0.1 port=5001 num-buffers=6 ! other/tensors,num_tensors=1,dimensions=3:300:300:1,types=uint8 ! tensor_query_serversink host=127.0.0.1 port=5002" 2-1 0 0 $PERFORMANCE $TIMEOUT_SEC &
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! tee name = t t. ! queue ! multifilesink location= raw2_%1d.log t. ! queue ! tensor_query_client src-host=127.0.0.1 src-port=5001 sink-host=127.0.0.1 sink-port=5002 ! multifilesink location=result2_%1d.log" 2-2 0 0 $PERFORMANCE &
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! tee name = t t. ! queue ! multifilesink location= raw2_2_%1d.log t. ! queue ! tensor_query_client src-host=127.0.0.1 src-port=5001 sink-host=127.0.0.1 sink-port=5002 ! multifilesink location=result2_2_%1d.log" 2-3 0 0 $PERFORMANCE
callCompareTest raw2_0.log result2_0.log 2-4 "Compare 2-4" 1 0
callCompareTest raw2_1.log result2_1.log 2-5 "Compare 2-5" 1 0
callCompareTest raw2_2.log result2_2.log 2-6 "Compare 2-6" 1 0
callCompareTest raw2_2_0.log result2_2_0.log 2-7 "Compare 2-7" 1 0
callCompareTest raw2_2_1.log result2_2_1.log 2-8 "Compare 2-8" 1 0
callCompareTest raw2_2_2.log result2_2_2.log 2-9 "Compare 2-9" 1 0

# Test flexible tensors
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc num-buffers=3 ! other/tensors,format=flexible ! tensor_query_serversink" 3-1 0 0 $PERFORMANCE $TIMEOUT_SEC &
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! other/tensors,format=flexible ! tee name = t t. ! queue ! multifilesink location= raw3_%1d.log t. ! queue ! tensor_query_client ! multifilesink location=result3_%1d.log" 3-2 0 0 $PERFORMANCE
callCompareTest raw3_0.log result3_0.log 3-3 "Compare 3-3" 1 0
callCompareTest raw3_1.log result3_1.log 3-4 "Compare 3-4" 1 0
callCompareTest raw3_2.log result3_2.log 3-5 "Compare 3-5" 1 0

# Check whether mqtt broker is running or not
pid=`ps aux | grep mosquitto | grep -v grep | awk '{print $2}'`
if [ $pid > 0 ]; then
    echo "mosquitto broker is running"
else
    echo "Warning! mosquitto is not running so skip MQTT-hybrid test."
    rm *.log
    report
    exit
fi

# Test MQTT-hybrid. Get server info using MQTT-hybrid
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc num-buffers=3 operation=passthrough ! other/tensors,format=flexible ! tee name = t t. ! queue ! multifilesink location=server1_%1d.log t. ! queue ! tensor_query_serversink" 4-1 0 0 $PERFORMANCE $TIMEOUT_SEC &
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc num-buffers=3 operation=passthrough port=5000 ! other/tensors,format=flexible ! tee name = t t. ! queue ! multifilesink location=server2_%1d.log t. ! queue ! tensor_query_serversink port=5001" 4-2 0 0 $PERFORMANCE $TIMEOUT_SEC &
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=7 ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! other/tensors,format=flexible ! tee name = t t. ! queue ! multifilesink location= raw4_%1d.log t. ! queue ! tensor_query_client operation=passthrough ! multifilesink location=result4_%1d.log" 4-3 0 0 $PERFORMANCE
callCompareTest raw4_0.log result4_0.log 4-4 "Compare 4-4" 1 0
callCompareTest raw4_1.log result4_1.log 4-5 "Compare 4-5" 1 0
callCompareTest raw4_2.log result4_2.log 4-6 "Compare 4-6" 1 0
# Server 1 is stopped and lost the fourth buffer.
callCompareTest raw4_4.log result4_3.log 4-7 "Compare 4-7" 1 0
callCompareTest raw4_5.log result4_4.log 4-8 "Compare 4-8" 1 0
callCompareTest raw4_6.log result4_5.log 4-9 "Compare 4-9" 1 0

# Compare the results of the server and the client.
callCompareTest server1_0.log result4_0.log 4-10 "Compare 4-10" 1 0
callCompareTest server1_1.log result4_1.log 4-11 "Compare 4-11" 1 0
callCompareTest server1_2.log result4_2.log 4-12 "Compare 4-12" 1 0
callCompareTest server2_0.log result4_3.log 4-13 "Compare 4-13" 1 0
callCompareTest server2_1.log result4_4.log 4-14 "Compare 4-14" 1 0
callCompareTest server2_2.log result4_5.log 4-15 "Compare 4-15" 1 0

rm *.log

report
