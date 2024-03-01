#!/usr/bin/env bash

##
# Imported from TAOS-CI
# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved.
# Modified for Github-Action in 2024.
#
# Argument 1 ($1): the number of commits in this pull-request

##
# @file     nobody.sh
# @brief    Check the commit message body. Originally, pr-prebuild-nobody.sh
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

for i in $(seq 0 $[$1 - 1]); do
  wordcount=`git show --pretty="format:%b" --no-notes -s HEAD~$i | wc -w`

  if [[ $wordcount -lt 8 ]]; then
    git show --stat HEAD~$i
    echo "::error The commit has too short commit message."
    exit 1
  fi
done
