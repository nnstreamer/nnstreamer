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
gstTest "filesrc location=video_4x4xBGRx.xraw ! application/octet-stream ! tensor_converter input-dim=4:4:4:1 input-type=uint8 ! fakesink" 1R 1 0 0

report
