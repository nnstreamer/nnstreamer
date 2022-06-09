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

# Skip query test temporarily because of ppa build failure.
# Related issue: https://github.com/nnstreamer/nnstreamer/issues/3657
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
if [ "$ID" = "ubuntu" ]; then
    echo "Skip query test for ppa build"
    report
    exit
fi

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

function waitformarker {
  for i in $(seq 1 ${TIMEOUT_SEC})
  do
    if [ -f 'marker.log' ]; then
      markersize=$(stat -c%s marker.log)
      if [ $markersize -ge 48 ]; then
        testResult 1 $1 "$2" 0 0
        return 0
      fi
    fi
    sleep 1
  done
  testResult 0 $1 "$2" 0 0
  exit
}
WAIT4MARKER=" videotestsrc num-buffers=1 ! video/x-raw,format=RGB,height=4,width=4 ! filesink location=marker.log "

# Run tensor query server as echo server with default address option.
rm -f marker.log
gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc num-buffers=3 ! other/tensors,num_tensors=1,dimensions=3:300:300:1,types=uint8 ! tensor_query_serversink async=false ${WAIT4MARKER} &
pid=$!
waitformarker 1-1-T "query-server launching"
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! tee name = t t. ! queue ! multifilesink location= raw1_%1d.log t. ! queue ! tensor_query_client ! multifilesink location=result1_%1d.log" 1-2 0 0 $PERFORMANCE
callCompareTest raw1_0.log result1_0.log 1-3 "Compare 1-3" 1 0
callCompareTest raw1_1.log result1_1.log 1-4 "Compare 1-4" 1 0
callCompareTest raw1_2.log result1_2.log 1-5 "Compare 1-5" 1 0
# Since the server operates in the background, wait for the server to stop before starting the next test.
kill -9 $pid &> /dev/null
wait $pid


# Run tensor query server as echo server with given address option. (multi clients)
rm marker.log
PORT1=`python3 ../get_available_port.py`
PORT2=`python3 ../get_available_port.py`
gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc host=127.0.0.1 port=${PORT1} num-buffers=6 ! other/tensors,num_tensors=1,dimensions=3:300:300:1,types=uint8 ! tensor_query_serversink async=false host=127.0.0.1 port=${PORT2} ${WAIT4MARKER} &
pid=$!
waitformarker 2-1-T "query-server launching"
rm marker.log
gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! tee name = t t. ! queue ! multifilesink location= raw2_%1d.log t. ! queue ! tensor_query_client src-host=127.0.0.1 src-port=${PORT1} sink-host=127.0.0.1 sink-port=${PORT2} ! multifilesink location=result2_%1d.log ${WAIT4MARKER} &
pid2=$!
waitformarker 2-2-T "query-client 1 launching"
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! tee name = t t. ! queue ! multifilesink location= raw2_2_%1d.log t. ! queue ! tensor_query_client src-host=127.0.0.1 src-port=${PORT1} sink-host=127.0.0.1 sink-port=${PORT2} ! multifilesink location=result2_2_%1d.log" 2-3 0 0 $PERFORMANCE
callCompareTest raw2_0.log result2_0.log 2-4 "Compare 2-4" 1 0
callCompareTest raw2_1.log result2_1.log 2-5 "Compare 2-5" 1 0
callCompareTest raw2_2.log result2_2.log 2-6 "Compare 2-6" 1 0
callCompareTest raw2_2_0.log result2_2_0.log 2-7 "Compare 2-7" 1 0
callCompareTest raw2_2_1.log result2_2_1.log 2-8 "Compare 2-8" 1 0
callCompareTest raw2_2_2.log result2_2_2.log 2-9 "Compare 2-9" 1 0
kill -9 $pid &> /dev/null
kill -9 $pid2 &> /dev/null
wait $pid
wait $pid2


# Test flexible tensors
rm marker.log
gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc num-buffers=3 ! other/tensors,format=flexible ! tensor_query_serversink async=false ${WAIT4MARKER} &
pid=$!
waitformarker 3-1-T "query-server launching"
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! other/tensors,format=flexible ! tee name = t t. ! queue ! multifilesink location= raw3_%1d.log t. ! queue ! tensor_query_client ! multifilesink location=result3_%1d.log" 3-2 0 0 $PERFORMANCE
callCompareTest raw3_0.log result3_0.log 3-3 "Compare 3-3" 1 0
callCompareTest raw3_1.log result3_1.log 3-4 "Compare 3-4" 1 0
callCompareTest raw3_2.log result3_2.log 3-5 "Compare 3-5" 1 0
kill -9 $pid &> /dev/null
wait $pid

# Test multiple query server src and sink.
rm marker.log
PORT1=`python3 ../get_available_port.py`
PORT2=`python3 ../get_available_port.py`
gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} \
    tensor_query_serversrc id=0 num-buffers=3 ! other/tensors,format=flexible ! tensor_query_serversink id=0 async=false \
    tensor_query_serversrc id=1 port=${PORT1} num-buffers=3 ! other/tensors,format=flexible ! tensor_query_serversink id=1 port=${PORT2} async=false ${WAIT4MARKER} &
pid=$!
waitformarker 5-1-T "query-server launching"
# Client pipeline 5-2 is connected to server ID 0.
rm marker.log
gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} \
    videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,width=640,height=480,format=RGB ! \
    tensor_converter ! other/tensors,format=flexible ! tee name=t \
        t. ! queue ! multifilesink location= raw5_2_%1d.log \
        t. ! queue ! tensor_query_client ! multifilesink location=result5_2_%1d.log ${WAIT4MARKER} &
pid2=$!
waitformarker 5-2-T "query-client 1 launching"
# Client pipeline 5-3 is connected to server ID 1.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    videotestsrc pattern=13 num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! \
    tensor_converter ! other/tensors,format=flexible ! tee name=t \
        t. ! queue ! multifilesink location= raw5_3_%1d.log \
        t. ! queue ! tensor_query_client src-port=${PORT1} sink-port=${PORT2} ! multifilesink location=result5_3_%1d.log" 5-3 0 0 $PERFORMANCE
callCompareTest raw5_2_0.log result5_2_0.log 5-4 "Compare 5-4" 1 0
callCompareTest raw5_2_1.log result5_2_1.log 5-5 "Compare 5-5" 1 0
callCompareTest raw5_2_2.log result5_2_2.log 5-6 "Compare 5-6" 1 0
callCompareTest raw5_3_0.log result5_3_0.log 5-7 "Compare 5-7" 1 0
callCompareTest raw5_3_1.log result5_3_1.log 5-8 "Compare 5-8" 1 0
callCompareTest raw5_3_2.log result5_3_2.log 5-9 "Compare 5-9" 1 0
kill -9 $pid &> /dev/null
kill -9 $pid2 &> /dev/null
wait $pid
wait $pid2


# Sever src cap: Video, Server sink cap: Viedo test
rm marker.log
gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc num-buffers=3 ! video/x-raw,width=300,height=300,format=RGB,framerate=0/1 ! tensor_query_serversink async=false ${WAIT4MARKER} &
pid=$!
waitformarker 6-1-T "query-server launching"
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tee name = t t. ! queue ! multifilesink location= raw6_%1d.log t. ! queue ! tensor_query_client ! multifilesink location=result6_%1d.log" 6-2 0 0 $PERFORMANCE
callCompareTest raw6_0.log result6_0.log 6-3 "Compare 6-3" 1 0
callCompareTest raw6_1.log result6_1.log 6-4 "Compare 6-4" 1 0
callCompareTest raw6_2.log result6_2.log 6-5 "Compare 6-5" 1 0
kill -9 $pid &> /dev/null
wait $pid


# Sever src cap: Video, Server sink cap: Tensor test
rm marker.log
gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc num-buffers=3 ! video/x-raw,width=300,height=300,format=RGB,framerate=0/1 ! tensor_converter ! tensor_query_serversink async=false ${WAIT4MARKER} &
pid=$!
waitformarker 7-1-T "query-server launching"
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tee name = t t. ! queue ! multifilesink location= raw7_%1d.log t. ! queue ! tensor_query_client ! multifilesink location=result7_%1d.log" 7-2 0 0 $PERFORMANCE
callCompareTest raw7_0.log result7_0.log 7-3 "Compare 7-3" 1 0
callCompareTest raw7_1.log result7_1.log 7-4 "Compare 7-4" 1 0
callCompareTest raw7_2.log result7_2.log 7-5 "Compare 7-5" 1 0
kill -9 $pid &> /dev/null
wait $pid


# Sever src cap: Tensor, Server sink cap: Video test
rm marker.log
gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} tensor_query_serversrc num-buffers=3 ! other/tensors,num_tensors=1,dimensions=3:300:300:1,types=uint8,format=static,framerate=0/1 ! tensor_decoder mode=direct_video ! videoconvert ! tensor_query_serversink async=false ${WAIT4MARKER} &
pid=$!
waitformarker 8-1-T "query-server launching"
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! tee name = t t. ! queue ! multifilesink location= raw8_%1d.log t. ! queue ! tensor_query_client ! multifilesink location=result8_%1d.log" 8-2 0 0 $PERFORMANCE
callCompareTest raw8_0.log result8_0.log 8-3 "Compare 8-3" 1 0
callCompareTest raw8_1.log result8_1.log 8-4 "Compare 8-4" 1 0
callCompareTest raw8_2.log result8_2.log 8-5 "Compare 8-5" 1 0
kill -9 $pid &> /dev/null
wait $pid


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
sleep $SLEEPTIME_SEC
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

sleep $SLEEPTIME_SEC
rm *.log

report
