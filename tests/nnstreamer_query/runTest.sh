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

# Run tensor query server as echo server with default address option.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc num-buffers=3 ! other/tensors,num_tensors=1,dimensions=3:300:300:1,types=uint8 ! tensor_query_serversink" 1-1 0 0 $PERFORMANCE $TIMEOUT_SEC &
sleep 1
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! tee name = t t. ! queue ! multifilesink location= raw1_%1d.log t. ! queue ! tensor_query_client ! multifilesink location=result1_%1d.log" 1-2 0 0 $PERFORMANCE
callCompareTest raw1_0.log result1_0.log 1-3 "Compare 1-3" 1 0
callCompareTest raw1_1.log result1_1.log 1-4 "Compare 1-4" 1 0
callCompareTest raw1_2.log result1_2.log 1-5 "Compare 1-5" 1 0

# Since the server operates in the background, wait for the server to stop before starting the next test.
sleep 2

# Run tensor query server as echo server with given address option. (multi clients)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc host=127.0.0.1 port=5001 num-buffers=6 ! other/tensors,num_tensors=1,dimensions=3:300:300:1,types=uint8 ! tensor_query_serversink host=127.0.0.1 port=5002" 2-1 0 0 $PERFORMANCE $TIMEOUT_SEC &
sleep 1
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! tee name = t t. ! queue ! multifilesink location= raw2_%1d.log t. ! queue ! tensor_query_client src-host=127.0.0.1 src-port=5001 sink-host=127.0.0.1 sink-port=5002 ! multifilesink location=result2_%1d.log" 2-2 0 0 $PERFORMANCE &
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! tee name = t t. ! queue ! multifilesink location= raw2_2_%1d.log t. ! queue ! tensor_query_client src-host=127.0.0.1 src-port=5001 sink-host=127.0.0.1 sink-port=5002 ! multifilesink location=result2_2_%1d.log" 2-3 0 0 $PERFORMANCE
callCompareTest raw2_0.log result2_0.log 2-4 "Compare 2-4" 1 0
callCompareTest raw2_1.log result2_1.log 2-5 "Compare 2-5" 1 0
callCompareTest raw2_2.log result2_2.log 2-6 "Compare 2-6" 1 0
callCompareTest raw2_2_0.log result2_2_0.log 2-7 "Compare 2-7" 1 0
callCompareTest raw2_2_1.log result2_2_1.log 2-8 "Compare 2-8" 1 0
callCompareTest raw2_2_2.log result2_2_2.log 2-9 "Compare 2-9" 1 0

sleep 2

# Test flexible tensors
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc num-buffers=3 ! other/tensors,format=flexible ! tensor_query_serversink" 3-1 0 0 $PERFORMANCE $TIMEOUT_SEC &
sleep 1
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! other/tensors,format=flexible ! tee name = t t. ! queue ! multifilesink location= raw3_%1d.log t. ! queue ! tensor_query_client ! multifilesink location=result3_%1d.log" 3-2 0 0 $PERFORMANCE
callCompareTest raw3_0.log result3_0.log 3-3 "Compare 3-3" 1 0
callCompareTest raw3_1.log result3_1.log 3-4 "Compare 3-4" 1 0
callCompareTest raw3_2.log result3_2.log 3-5 "Compare 3-5" 1 0

sleep 2

# Test multiple query server src and sink.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    tensor_query_serversrc id=0 num-buffers=3 ! other/tensors,format=flexible ! tensor_query_serversink id=0 async=false \
    tensor_query_serversrc id=1 port=5000 num-buffers=3 ! other/tensors,format=flexible ! tensor_query_serversink id=1 port=5001  async=false" 5-1 0 0 $PERFORMANCE $TIMEOUT_SEC &
sleep 1
# Client pipeline 5-2 is connected to server ID 0.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,width=640,height=480,format=RGB ! \
    tensor_converter ! other/tensors,format=flexible ! tee name=t \
        t. ! queue ! multifilesink location= raw5_2_%1d.log \
        t. ! queue ! tensor_query_client ! multifilesink location=result5_2_%1d.log" 5-2 0 0 $PERFORMANCE &
# Client pipeline 5-3 is connected to server ID 1.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    videotestsrc pattern=13 num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! \
    tensor_converter ! other/tensors,format=flexible ! tee name=t \
        t. ! queue ! multifilesink location= raw5_3_%1d.log \
        t. ! queue ! tensor_query_client src-port=5000 sink-port=5001 ! multifilesink location=result5_3_%1d.log" 5-3 0 0 $PERFORMANCE
callCompareTest raw5_2_0.log result5_2_0.log 5-4 "Compare 5-4" 1 0
callCompareTest raw5_2_1.log result5_2_1.log 5-5 "Compare 5-5" 1 0
callCompareTest raw5_2_2.log result5_2_2.log 5-6 "Compare 5-6" 1 0
callCompareTest raw5_3_0.log result5_3_0.log 5-7 "Compare 5-7" 1 0
callCompareTest raw5_3_1.log result5_3_1.log 5-8 "Compare 5-8" 1 0
callCompareTest raw5_3_2.log result5_3_2.log 5-9 "Compare 5-9" 1 0

sleep 2

# Sever src cap: Video, Server sink cap: Viedo test
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc num-buffers=3 ! video/x-raw,width=300,height=300,format=RGB,framerate=0/1 ! tensor_query_serversink" 6-1 0 0 $PERFORMANCE $TIMEOUT_SEC &
sleep 1
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tee name = t t. ! queue ! multifilesink location= raw6_%1d.log t. ! queue ! tensor_query_client ! multifilesink location=result6_%1d.log" 6-2 0 0 $PERFORMANCE
callCompareTest raw6_0.log result6_0.log 6-3 "Compare 6-3" 1 0
callCompareTest raw6_1.log result6_1.log 6-4 "Compare 6-4" 1 0
callCompareTest raw6_2.log result6_2.log 6-5 "Compare 6-5" 1 0

sleep 2

# Sever src cap: Video, Server sink cap: Tensor test
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc num-buffers=3 ! video/x-raw,width=300,height=300,format=RGB,framerate=0/1 ! tensor_converter ! tensor_query_serversink" 7-1 0 0 $PERFORMANCE $TIMEOUT_SEC &
sleep 1
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tee name = t t. ! queue ! multifilesink location= raw7_%1d.log t. ! queue ! tensor_query_client ! multifilesink location=result7_%1d.log" 7-2 0 0 $PERFORMANCE
callCompareTest raw7_0.log result7_0.log 7-3 "Compare 7-3" 1 0
callCompareTest raw7_1.log result7_1.log 7-4 "Compare 7-4" 1 0
callCompareTest raw7_2.log result7_2.log 7-5 "Compare 7-5" 1 0

sleep 2

# Sever src cap: Tensor, Server sink cap: Video test
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc num-buffers=3 ! other/tensors,num_tensors=1,dimensions=3:300:300:1,types=uint8,format=static,framerate=0/1 ! tensor_decoder mode=direct_video ! videoconvert ! tensor_query_serversink" 8-1 0 0 $PERFORMANCE $TIMEOUT_SEC &
sleep 1
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! tee name = t t. ! queue ! multifilesink location= raw8_%1d.log t. ! queue ! tensor_query_client ! multifilesink location=result8_%1d.log" 8-2 0 0 $PERFORMANCE
callCompareTest raw8_0.log result8_0.log 8-3 "Compare 8-3" 1 0
callCompareTest raw8_1.log result8_1.log 8-4 "Compare 8-4" 1 0
callCompareTest raw8_2.log result8_2.log 8-5 "Compare 8-5" 1 0

sleep 2

# TODO enable query-hybrid test
# Now nnsquery library is not available.
# After publishing the nnsquery pkg, enable below testcases.
rm *.log
report
exit

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
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc num-buffers=3 operation=passthrough ! other/tensors,format=flexible ! tee name = t t. ! queue ! multifilesink location=server1_%1d.log t. ! queue ! tensor_query_serversink" 4-1 0 0 $PERFORMANCE $TIMEOUT_SEC &
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc num-buffers=3 operation=passthrough port=5000 ! other/tensors,format=flexible ! tee name = t t. ! queue ! multifilesink location=server2_%1d.log t. ! queue ! tensor_query_serversink port=5001" 4-2 0 0 $PERFORMANCE $TIMEOUT_SEC &
sleep 1
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

sleep 2
rm *.log

report
