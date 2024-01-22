#!/usr/bin/env bash

##
# Imported from TAOS-CI
# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved.
# Modified for Github-Action in 2024.
#
# Argument 1 ($1): the file containing the list of files to be checked
#

##
# @file     hardcoded-path.sh
# @brief    Check prohibited hardcoded paths (/home/* for now). Originally pr-prebuild-hardcoded-path.sh
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
check grep
check wc

errorreport=$(mktemp)

for file in `cat $files`; do
  if [[ $file =~ \.cpp$ || $file =~ \.c$ || $file =~ \.hpp$ || $file =~ \.h$ ]]; then
    count=`grep "\"\/home\/" $file | wc -l`
    if [[ $VIOLATION -gt 0 ]]; then
      failed=1
      errstr=`grep "\"\/home\/" $file`
      echo "At file $file, found a hardcoded path:$errstr\n" >> $errorreport
    fi
  fi
done

if [[ "failed" = "1" ]]; then
  echo "::group::Files with hardcoded paths are:"
  cat $errorreport
  echo "::endgroup::"
  echo "::error A source code should not have a personal path hardcoded."
  exit 1
fi
