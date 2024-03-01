#!/usr/bin/env bash

##
# Imported from TAOS-CI
# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved.
# Modified for Github-Action in 2024.
#
# Argument 1 ($1): the file containing the list of files to be checked

##
# @file newline.sh
# @brief Check if there is a newline issue in a text file. Originally pr-prebuild-newline.sh
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

echo "::group::Newline check started"

num=0
report=$(mktemp)
finalreport=$(mktemp)
for file in `cat $files`; do
  newline_count=0

  # If a file is "ASCII text" type, let's check a newline rule.
  if [[ `file $file | grep "ASCII text" | wc -l` -gt 0 ]]; then
    num=$(( $num + 1 ))
    # fetch patch content of a specified file from  a commit.
    git show $file> ${report}.${num}.patch
    # check if the last line of a patch file includes "\ No newline....." statement.
    newline_count=$(cat ${report}.${num}.patch  | tail -1 | grep '^\\ No newline' | wc -l)
    if  [[ $newline_count == 0 ]]; then
      echo "$file is ok."
    else
      echo "$file has newline style error."
      failed=1
      echo "=========================================" >> $finalreport
      echo "$file has newline style error." >> $finalreport
      cat $report.$num.patch >> $finalreport
      echo "\n\n" >> $finalreport
    fi
  fi
done

echo "::endgroup::"

if [ $failed = 1 ]; then
    echo "::error There is a newline style error."
    echo "::group::The errors are..."
    cat $finalreport
    echo "::endgroup::"
    exit 1
fi
