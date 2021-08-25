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

rm *.log

report
