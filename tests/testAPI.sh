#!/usr/bin/env bash
if [[ $# -eq 0 ]]; then
	PATH_TO_PLUGIN="$PWD/../build/tensor_converter:$PWD/../build/tensor_filter:$PWD/../build/tensor_decoder"
else
	PATH_TO_PLUGIN="$1"
fi

RED='\033[1;31m'
GREEN='\033[1;32m'
BLUE='\033[1;34m'
PURPLE='\033[1;36m'
NC='\033[0m'


declare -i lsucc=0
declare -i lfail=0
log=""

##
# @brief execute gst-launch based test case
# @param 1 the whole parameters
# @param 2 the case number
function gstTest {
	stdout=$(gst-launch-1.0 -q $1)
	output=$?
	if [[ $output -eq 0 ]]; then
		lsucc=$((lsucc+1))
		log="${log}$GREEN[PASSED]$NC gst-launch with case $2\n"
	else
		printf "$stdout\n"
		lfail=$((lfail+1))
		log="${log}$RED[FAILED] gst-launch with case $2$NC\n"
	fi
}

##
# @brief compare the golden output and the actual output
# @param 1 the golden case file
# @param 2 the actual output
# @param 3 the case number
function compareAll {
	cmp $1 $2
	output=$?
	if [[ $output -eq 0 ]]; then
		lsucc=$((lsucc+1))
		log="${log}$GREEN[PASSED]$NC golden test comparison case $3\n"
	else
		lfail=$((lfail+1))
		log="${log}$RED[FAILED] golden test comparison case $3$NC\n"
	fi
}


##
# @brief compare the golden output and the actual output with the size limit on golden output.
# @param 1 the golden case file
# @param 2 the actual output
# @param 3 the case number
function compareAllSizeLimit {
	# @TODO enter -n option with the size of #1
	cmp -n `stat --printf="%s" $1` $1 $2
	if [[ $output -eq 0 ]]; then
		lsucc=$((lsucc+1))
		log="${log}$GREEN[PASSED]$NC golden test comparison case $3\n"
	else
		lfail=$((lfail+1))
		log="${log}$RED[FAILED] golden test comparison case $3$NC\n"
	fi
}

##
# @brief "runTest.sh" should call this at the end of the file.
function report {
	printf "${log}"
	printf "${lsucc} ${lfail}\n"
	if [[ $lfail -eq 0 ]]; then
		exit 0
	else
		exit 1
	fi
}
