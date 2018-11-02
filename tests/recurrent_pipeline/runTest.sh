#!/usr/bin/env bash
##
## @file runTest.sh
## @author MyungJoo Ham <myungjoo.ham@gmail.com>
## @date Nov 01 2018
## @brief This is a template file for SSAT test cases. You may designate your own license.
##
if [[ "$SSATAPILOADED" != "1" ]]
then
	SILENT=0
	INDEPENDENT=1
	search="ssat-api.sh"
	source $search
	printf "${Blue}Independent Mode${NC}
"
fi
testInit $1 # You may replace this with Test Group Name

# Generate video_4x4xBGRx.xraw
python generateTestCase.py

# This test case is WIP. So do not fail unittest even if this case fails.

PATH_TO_MODEL=../../build/nnstreamer_example/custom_example_LSTM/libdummyLSTM.so
# Testing the dummy LSTM without recurrences.
gstTest "filesrc location=video_4x4xBGRx.xraw ! application/octet-stream ! \
tensor_converter input-dim=4:4:4:1 input-type=uint8 ! tee name=t \
t. ! queue ! mux.sink_0  \
t. ! queue ! mux.sink_1 \
tensor_mux name=mux ! tensor_filter framework=\"custom\" model=\"${PATH_TO_MODEL}\" !
filesink location=\"testcase01.doublein.log\" sync=true " 1NR 1 0 0

## @todo Do the golden test with testcase01.doublein.log
# Testing with recurrence. This goes into infinite loop. Don't do this until we have added "LSTM" mode to tensor_mux.
gstTest "INDUCEERROR ! filesrc location=video_4x4xBGRx.xraw ! application/octet-stream ! \
tensor_converter input-dim=4:4:4:1 input-type=uint8 ! mux.sink_0 \
t. ! queue ! mux.sink_1 \
tensor_mux name=mux ! tensor_filter framework=\"custom\" model=\"${PATH_TO_MODEL}\" ! \
tee name=t \
t. ! queue ! filesink location=\"testcase01.doublein.log\" sync=true " 2R 1 0 0


report
