#!/usr/bin/env bash

display_help() {
    echo -e "Usage: $0 [option...]" >&2
    echo -e ""
    echo -e "   1       Generate GStreamer Pipeline Graphs"
    echo -e "           with GST_DEBUG_DUMP_DOT_DIR"
    echo -e ""
    exit 1
}

case $1 in
	-h) display_help ;;
	--help) display_help ;;
esac

dirpath="$( cd "$( dirname "$0")" && pwd )"

if [[ $1 -eq 1 ]]; then
	export GST_DEBUG_DUMP_DOT_DIR=$dirpath/performance
	export LD_PRELOAD=/usr/local/lib/libgstintercept.so
	export GST_DEBUG_DUMP_TRACE_DIR=$GST_DEBUG_DUMP_DOT_DIR
	export PERFORMANCE=1
	mkdir -p $GST_DEBUG_DUMP_DOT_DIR
fi
shift 1
source $dirpath/testAPI.sh 

if [[ $PERFORMANCE -eq 1 ]]; then
	checkDependency dot
	checkDependency gst-report-1.0
fi
checkDependency dirname
checkDependency basename
checkDependency sed
checkDependency find

sopath=""
log=""
summary=""
retval=0

declare -i success=0
declare -i failed=0

while IFD= read -r -d $'\0' line; do
	dir=$(dirname "${line}")
	export base=$(basename ${dir})

	if [[ $PERFORMANCE -eq 1 ]]; then
		mkdir -p $GST_DEBUG_DUMP_DOT_DIR/debug/$base
		mkdir -p $GST_DEBUG_DUMP_DOT_DIR/profile/$base
	fi

	log="${log}=================================================\n"
	log="${log}${BLUE}Testing${NC}: ${PURPLE}${base}${NC}\n"
	log="${log}=================================================\n"
	pushd $dir > /dev/null
	result=`./runTest.sh ${PATH_TO_PLUGIN}`
	result="$result\n"
	lastline="${result##*$'\n'}"
	showline=$(printf "$result" | sed '$d')
	showline="${showline}\n"
	log="${log}${showline}"
	lsucc=$(printf "${lastline}" | sed 's|^\([0-9][0-9]*\) .*$|\1|')
	lfail=$(printf "${lastline}" | sed 's|^.* \([0-9][0-9]*\)$|\1|')
	ltotal=$((lsucc+lfail))
	if [[ $lfail -eq 0 ]]; then
		success=$((success + lsucc))
		summary="${summary}$GREEN[PASSED] ${PURPLE}$base ${BLUE}(${lsucc} of ${ltotal} passed)${NC}\n"
	else
		failed=$((failed + lfail))
		success=$((success + lsucc))
		summary="${summary}$RED[FAILED] ${PURPLE}$base ${BLUE}(${lsucc} Passed / ${lfail} Failed of ${ltotal} Cases)${NC}\n"
	fi
	popd > /dev/null
done < <(find $dirpath -name "runTest.sh" -print0)

printf "\n\n\n"
printf "$log"
printf "\n\n=================================================\n"
printf "$summary"
printf "\n\n=================================================\n"
total=$((success+failed))
printf "$GREEN[PASSED]$NC $success (Total = $total)\n"
if [[ $failed -eq 0 ]]; then
	printf "$GREEN[PASSED] All Testcase Passed$NC\n"
	retval=0
else
	printf "$RED[FAILED]$NC ${PURPLE}$failed cases failed.${NC}\n"
	retval=1
fi
printf "=================================================\n"
printf "\n\n"


exit $retval
