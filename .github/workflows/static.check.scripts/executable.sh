#!/usr/bin/env bash

##
# Imported from TAOS-CI
# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved.
# Modified for Github-Action in 2024.
#
# Argument 1 ($1): the list of files changed in this pull request.

##
# @file     executable.sh
# @brief    Check executable bits for .cpp, .c, .hpp, .h, .prototxt, .caffemodel, .txt., .init
#           Originally, pr-prebuild-executable.sh
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

errorreport=$(mktemp)

for file in `cat $files`; do
  if [[ $file =~ \.cpp$ || $file =~ \.c$ || $file =~ \.hpp$ || $file =~ \.h$ || $file =~ \.prototxt$ || $file =~ \.caffemodel$ || $file =~ \.txt$ || $file =~ \.ini$ ]]; then
    if [[ -f "$file" && -x "$file" ]]; then
      # It is a text file (.cpp, .c, ...) and is executable. This is invalid!
      failed=1
      ls -la $file >> $errorreport
    fi
  fi
done

if [ $failed = 1 ]; then
  echo "::group::List of files with executable bits that should not be executable"
  cat $errorreport
  echo "::endgroup::"
  echo "::error There are source code files with executable bits. Fix them."
  exit 1
fi
