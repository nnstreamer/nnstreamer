#!/usr/bin/env bash

##
# Imported from TAOS-CI
# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved.
# Modified for Github-Action in 2024.
#
# Argument 1 ($1): the file containing the list of files to be checked
# Argument 2 ($2): the flawfinder check level. (1 by default)
#

##
# @file     flawfinder.sh
# @brief    This module examines C/C++ source code to find a possible security weaknesses.
#           Originally, pr-prebuild-flawfinder.sh
#
#  Flawfinder searches through C/C++ source code looking for potential security flaws. It examines
#  all of the project's C/C++ source code. It is very useful for quickly finding and removing some
#  security problems before a program is widely released.
#
#  Flawfinder produces a list of `hits` (potential security flaws), sorted by risk; the riskiest
#  hits are shown first. The risk level is shown inside square brackets and varies from 0 (very
#  little risk) to 5 (great risk). This risk level depends not only on the function,
#  but on the values of the parameters of the function. For example, constant strings are often less risky
#  than fully variable strings in many contexts, and in those contexts the hit will have a lower risk level.
#  Hit descriptions also note the relevant Common Weakness Enumeration (CWE) identifier(s) in parentheses.
#
# @see      https://dwheeler.com/flawfinder/
# @see      https://sourceforge.net/projects/flawfinder/
# @see      https://github.com/nnstreamer/TAOS-CI
# @see      https://github.com/nnstreamer/nnstreamer
# @author   Geunsik Lim <geunsik.lim@samsung.com>
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>

if [ -z $1 ]; then
  echo "::error The argument (file path) is not given."
  exit 1
fi

pr_flawfinder_check_level=1
if [ -z $2 ]; then
  echo "Flawfinder check level is not given. The default value 1 is applied"
else
  if [ -n "$2" ] && [ "$2" -eq "$2" ] 2>/dev/null; then
    echo ""
  else
    echo ":error The second argument '$2' should be a number."
    exit 1
  fi
  pr_flawfinder_check_level=$2
fi

files=$1
failed=0

if [ ! -f $files ]; then
  echo "::error The file $files does not exists."
  exit 1
fi

function check(){
  which $1
  if [[ $? -ne 0 ]]; then
    echo "::error The command $1 is required, but not found."
    exit 1
  fi
}
# Check if server administrator install required commands
check flawfinder
check file
check grep
check cat
check wc
check git
check awk

static_analysis_sw="flawfinder"
static_analysis_rules="--html --context --minlevel=$pr_flawfinder_check_level"
flawfinder_result=$(mktemp)

# Display the flawfinder version that is installed in the CI server.
# Note that the out-of-date version can generate an incorrect result.
flawfinder --version

# Read file names that a contributor modified (e.g., added, moved, deleted, and updated) from a last commit.
# Then, inspect C/C++ source code files from *.patch files of the last commit.
for file in `cat $file`; do
  # Skip the obsolete folder
  if [[ $file =~ ^obsolete/.* ]]; then
    continue
  fi
  # Skip the external folder
  if [[ $file =~ ^external/.* ]]; then
    continue
  fi
  # Handle only text files in case that there are lots of files in one commit.
  if [[ `file $file | grep "ASCII text" | wc -l` -gt 0 ]]; then
    # in case of C/C++ source code
    case $file in
      # in case of C/C++ code
      *.c|*.cc|*.cpp|*.c++)
        $static_analysis_sw $static_analysis_rules $file > ${flawfinder_result}
        bug_nums=`cat ${flawfinder_result} | grep "Hits = " | awk '{print $3}'`
        # Report the execution result.
        if  [[ $bug_nums -gt 0 ]]; then
            echo "[ERROR] $static_analysis_sw: failed. file name: $file, There are $bug_nums bug(s)."
            echo "::group::The flawfinder result of $file:"
            cat ${flawfinder_result}
            echo "::endgroup::"
            failed=1
        else
            echo "[DEBUG] $static_analysis_sw: passed. file name: $file, There are $bug_nums bug(s)."
        fi
        ;;
    esac
  fi
done

if [[ "$failed" == "1" ]]; then
  echo "::error There is an error from flawfinder. Please review the detailed results above."
  exit 1
fi
