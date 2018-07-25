#!/usr/bin/env bash

if [[ $# -eq 0 ]]; then
	dirpath="$( cd "$( dirname "$0")" && pwd )"
	find "$dirpath/../build/gst/tensor_converter" "$dirpath/../build/gst/tensor_filter" "$dirpath/../build/gst/tensor_decoder" -name *.so 1>/dev/null 2>/dev/null
	if [ "$?" -ne "0" ]; then
		dirpath="$dirpath/../"
		find "$dirpath/../build/gst/tensor_converter" "$dirpath/../build/gst/tensor_filter" "$dirpath/../build/gst/tensor_decoder" -name *.so 1>/dev/null 2>/dev/null
		if [ "$?" -ne "0" ]; then
			dirpath="$dirpath/../"
			find "$dirpath/../build/gst/tensor_converter" "$dirpath/../build/gst/tensor_filter" "$dirpath/../build/gst/tensor_decoder" -name *.so 1>/dev/null 2>/dev/null
			if [ "$?" -ne "0" ]; then
				echo "[ERROR] Cannot find nnstreamer plugin binaries. Before unit testing, you should build with cmake first."
				exit 1
			fi
		fi
	fi
	PATH_TO_PLUGIN="$dirpath/../build/gst/tensor_converter:$dirpath/../build/gst/tensor_filter:$dirpath/../build/gst/tensor_decoder:$dirpath/../build/gst/tensor_transform:$dirpath/../build/gst/tensor_sink:$dirpath/../build/gst/tensor_mux:$dirpath/../build/gst/tensor_demux"
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
# @brief check if a command is installed
# @param 1 the command name
function checkDependency {
    which "$1" 1>/dev/null || {
      echo "Ooops. '$1' command is not installed. Please install '$1'."
      exit 1
    }
}

checkDependency gst-launch-1.0
checkDependency cmp

gst-inspect-1.0 pngdec 1>/dev/null || {
	echo "The plugin 'pngdec' is currently not installed. You can install it by typing:"
	echo "sudo apt install gstreamer1.0-plugins-good"
	exit 1
}

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
# @brief execute gst-launch that is expected to fail. This may be used to test if the element fails gracefully.
# @param 1 the whole parameters
# @param 2 the case number
function gstFailTest {
	stdout=$(gst-launch-1.0 -q $1)
	output=$?
	if [[ $output -eq 0 ]]; then
		printf "$stdout\n"
		lfail=$((lfail+1))
		log="${log}$RED[FAILED] (Suppsoed to be failed, but passed) gst-launch with case $2$NC\n"
	else
		lsucc=$((lsucc+1))
		log="${log}$GREEN[PASSED]$NC (Supposed to be failed and actually failed) gst-launch with case $2\n"
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
	if [ ! -f $1 ]; then
		output=1
	else
		size1=$(wc -c < "$1")
		if [ $size1 -eq 0 ]; then
			# If the size is 0, it's an error
			output=2
		fi
	fi
	if [ ! -f $2 ]; then
		output=1
	else
		size2=$(wc -c < "$2")
		if [ $size2 -eq 0 ]; then
			# If the size is 0, it's an error
			output=2
		fi
	fi
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
	output=$?
	if [ ! -f $1 ]; then
		output=1
	else
		size1=$(wc -c < "$1")
		if [ $size1 -eq 0 ]; then
			# If the size is 0, it's an error
			output=2
		fi
	fi
	if [ ! -f $2 ]; then
		output=1
	else
		size2=$(wc -c < "$2")
		if [ $size2 -eq 0 ]; then
			# If the size is 0, it's an error
			output=2
		fi
	fi
	if [[ $output -eq 0 ]]; then
		lsucc=$((lsucc+1))
		log="${log}$GREEN[PASSED]$NC golden test comparison case $3\n"
	else
		lfail=$((lfail+1))
		log="${log}$RED[FAILED] golden test comparison case $3$NC\n"
	fi
}

##
# @brief Report additional script test case result
# @param 1 the case number
# @param 2 the result (exit code)
# @param 3 test description
function casereport {
	if [[ $2 -eq 0 ]]; then
		lsucc=$((lsucc+1))
		log="${log}$GREEN[PASSED]$NC $3: $1\n"
	else
		lfail=$((lfail+1))
		log="${log}$RED[FAILED] $3: $1$NC\n"
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

##
# @brief Convert all *.bmp to *.png
function convertBMP2PNG {
	tool="bmp2png"
	if [ -x bmp2png ]; then
		tool="bmp2png"
	else
		if [ -x ../bmp2png ]; then
			tool="../bmp2png"
		else
			if [ -x ../../bmp2png ]; then
				tool="../../bmp2png"
			else
				tool="../../../bmp2png"
				# Try this and die if fails
			fi
		fi
	fi
	for X in `ls *.bmp`
	do
		$tool $X
	done
}
