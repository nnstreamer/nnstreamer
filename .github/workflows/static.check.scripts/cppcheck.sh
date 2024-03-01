#!/usr/bin/env bash

##
# Imported from TAOS-CI
# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved.
# Modified for Github-Action in 2024.
#
# Argument 1 ($1): the file containing the list of files to be checked
# Argument 2 ($2): check level. (default 0)

##
# @file     cppcheck.sh
# @brief    Check dangerous coding constructs in source codes (*.c, *.cpp) with a cppcheck tool
#           Originally pr-prebuild-cppcheck.sh
#
# The possible severities (e.g., --enable=warning,unusedFunction) for messages are as following:
# Note that by default Cppcheck only writes error messages if it is certain.
# 1. error  : used when bugs are found
# 2. warning: suggestions about defensive programming to prevent bugs
# 3. style  : stylistic issues related to code cleanup (unused functions, redundant code, constness, and such)
# 4. performance: Suggestions for making the code faster. These suggestions are only based on common knowledge.
# 5. portability: portability warnings. 64-bit portability. code might work different on different compilers. etc.
# 6. information: Informational messages about checking problems.
# 7. unusedFunction: enable unusedFunction checking. This is not enabled by --enable=style
#    because it does not work well on libraries.
# 8. all: enable all messages. It should also only be used when the whole program is scanned.
#
# @see      https://github.com/nnstreamer/TAOS-CI
# @see      https://github.com/danmar/cppcheck
# @see      https://github.com/nnstreamer/nnstreamer
# @author   Geunsik Lim <geunsik.lim@samsung.com>
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>
#

function version(){
    echo "$@" | awk -F. '{ printf("%d%03d%03d%03d\n", $1,$2,$3,$4); }';
}

if [ -z $1 ]; then
  echo "::error The argument (file path) is not given."
  exit 1
fi
if [ -z $2 ]; then
  2=0
fi

function check(){
  which $1
  if [[ $? -ne 0 ]]; then
    echo "::error The command $1 is required, but not found."
    exit 1
  fi
}

check cppcheck
check file
check grep
check cat
check wc
check awk

files=$1
failed=0

if [ ! -f $files ]; then
  echo "::error The file $files does not exists."
  exit 1
fi

# Display the cppcheck version that is installed in the CI server.
# Note that the out-of-date version can generate an incorrect result.
cppcheck_ver=$(cppcheck --version | awk {'print $2'})
default_cmd="--std=posix"
# --std=posix is deprecated and removed in 2.0.5
if [[ $(version $cppcheck_ver) -ge $(version "2.0.5") ]]; then
    default_cmd="--library=posix"
fi

static_analysis_sw="cppcheck"

if [[ $2 -eq 0 ]]; then
  echo "cppcheck: Default mode."
  static_analysis_rules="$default_cmd"
elif [[ $2 -eq 1 ]]; then
  echo "cppcheck: --enable=warning,performance added."
  static_analysis_rules="--enable=warning,performance $default_cmd"
else
  echo "cppcheck: $2 is an incorrect optiona. Overriding it as 0"
  static_analysis_rules="$default_cmd"
fi

errfile=$(mktemp)
error="no"
errors=""

for file in `cat $files`; do
  # skip obsolete folder
  if [[ $file =~ ^obsolete/.* ]]; then
    continue
  fi
  # skip external folder
  if [[ $file =~ ^external/.* ]]; then
    continue
  fi
  # Handle only text files in case that there are lots of files in one commit.
  if [[ `file $file | grep "ASCII text" | wc -l` -gt 0 ]]; then
    # in case of source code files (*.c, *.cpp)
    case $file in
      # in case of C/C++ code
      *.c|*.cpp)
        echo "( $file ) file is source code with the text format."
        # Check C/C++ file, enable all checks.
        $static_analysis_sw $static_analysis_rules $file 2> $errfile
        bug_line=`$errfile | wc -l `
        if  [[ $bug_line -gt 0 ]]; then
          echo "$file cppcheck result shows some errors. There are $bug_line bug(s):"
          echo ":group:cppcheck result of $file"
          cat $errfile
          echo ":endgroup:"
          error="yes"
          errors+=" $file"
        else
          echo "$file cppcheck result is ok."
        fi
        ;;
    esac
  fi
done

if [[ "$error" == "yes" ]]; then
  echo "cppcheck shows errors in: $errors"
  exit 1
fi
