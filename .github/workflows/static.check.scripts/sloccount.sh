#!/usr/bin/env bash

##
# Imported from TAOS-CI
# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved.
# Modified for Github-Action in 2024.
#
# Argument 1 ($1): the file containing the list of files to be checked
#

##
# @file     sloccount.sh
# @brief    Count physical source lines of code (SLOC). Originally, pr-prebuild-sloccount.sh
#
# It is a set of tools for counting physical Source Lines of Code (SLOC) in a large
# number of languages of a potentially large set of programs.
#
# @see      https://packages.ubuntu.com/search?keywords=sloccount
# @see      https://github.com/nnstreamer/TAOS-CI
# @see      https://github.com/nnstreamer/nnstreamer
# @author   Geunsik Lim <geunsik.lim@samsung.com>
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>
#

if [ -z $1 ]; then
  echo "::error The argument (file path) is not given."
  exit 1
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
check sloccount
check git
check file
check mkdir
check grep

sloc_analysis_sw="sloccount"
sloc_data_folder=$(mktemp -d)
sloc_analysis_rules="--wide --multiproject --datadir $sloc_data_folder"
sloc_check_result=$(mktemp)

# Inspect all files that contributor modifed.
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
    # Run a SLOCCount module in case that a PR includes source codes.
    case $file in
      *.c | *.cpp | *.py | *.sh | *.php )
      sloc_target_dir=${SRC_PATH}

      # Run this module
      echo "::group::sloccount result"
      $sloc_analysis_sw $sloc_analysis_rules . > ${sloc_check_result}
      echo "::endgroup::"
      run_result=$?
      if [[ $run_result -eq 0 ]]; then
        echo ""
        exit 0
      else
        echo "::error Sloccount failed."
        exit 1
      fi
      ;;
    esac
  fi
done
