#!/usr/bin/env bash

##
# Imported from TAOS-CI
# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved.
# Modified for Github-Action in 2024.
#
# Argument 1 ($1): the number of commits in this PR
#

##
# @file     signed-off-by.sh
# @brief    Check if contributor write Sigend-off-by message. Originally, pr-prebuild-signed-off-by.sh
# @see      https://github.com/nnstreamer/TAOS-CI
# @see      https://github.com/nnstreamer/nnstreamer
# @author   Sewon oh <sewon.oh@samsung.com>
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

# Check if server administrator install required commands
check git

for i in $(seq 0 $[$1 - 1]); do
  count=`git show --no-notes -s HEAD~$i | grep "Signed-off-by: [^ ].*[^ ].*<[^ ].*@[^ ].*>" | wc -l`
  if [[ $count -lt 1 ]]; then
    git show --stat HEAD~$i
    echo "============================================"
    echo ""
    echo "::error This commit does not have a proper Signed-off-by. Refer to https://ltsi.linuxfoundation.org/software/signed-off-process/ for some information about Signed-off-by. We require Signed-off-by: NAME <EMAIL> signed by indivisual contributors."
    exit 1
  else
    id=`git show --pretty="format:%h" --no-notes -s HEAD~$i`
    echo "Commit $id has proper a signed-off-by string."
  fi
done
