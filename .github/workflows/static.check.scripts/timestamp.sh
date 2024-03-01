#!/usr/bin/env bash

##
# Imported from TAOS-CI
# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved.
# Modified for Github-Action in 2024.
#
# Argument 1 ($1): the number of commits in this pull-request

##
# @file     timestamp.sh
# @brief    Check the timestamp of the commit. Originally, pr-prebuild-timestamp.sh
# @see      https://github.com/nnstreamer/TAOS-CI
# @see      https://github.com/nnstreamer/nnstreamer
# @author   Geunsik Lim <geunsik.lim@samsung.com>
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>
#

if [ -z $1 ]; then
  echo "::error The argument (the number of commits in this PR) is not given."
  exit 1
fi

if [ -n "$1" ] && [ "$1" -eq "$1" ] 2>/dev/null; then
  echo ""
else
  echo ":error The first argument '$1' should be a number."
  exit 1
fi

function check(){
  which $1
  if [[ $? -ne 0 ]]; then
    echo "::error The command $1 is required, but not found."
    exit 1
  fi
}
check git
check seq

NOW=`date +%s`
NOW_READ=`date`

for i in $(seq 0 $[$1 - 1]); do
  timestamp=`git show --pretty="%ct" --no-notes -s HEAD~$i`
  timestamp_read=`git show --pretty="%cD" --no-notes -s HEAD~$i`
  timestamp_3min=$(( $timestamp - 180 ))
  # allow 3 minutes of clock drift.

  if [[ $timestamp_3min -gt $NOW ]]; then
    git show --stat HEAD~$i
    echo "::error The commit has timestamp error, coming from the future: $timestamp_read (now: $NOW_READ)."
    exit 1
  fi
done
