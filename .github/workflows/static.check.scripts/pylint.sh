#!/usr/bin/env bash

##
# Imported from TAOS-CI
# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved.
# Modified for Github-Action in 2024.
#
# Argument 1 ($1): the file containing the list of files to be checked

##
# @file pylint.sh
# @brief Fail on pylint errors (E/F messages) in Python files. Originally pr-prebuild-pylint.sh
# @see      https://www.pylint.org/
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

which pylint
if [[ $? -ne 0 ]]; then
  echo "::error The pylint utility is not found."
  exit 1
fi

echo "::group::Pylint check started"

result=$(mktemp)
errlog=$(mktemp)

for file in `cat $files`; do
  if [[ $file =~ ^obsolete/.* ]]; then
    continue
  fi
  if [[ $file =~ ^external/.* ]]; then
    continue
  fi

  if [[ `file $file | grep "ASCII text" | wc -l` -gt 0 ]]; then
    case $file in
      *.py)
        echo "Checking $file"
        # CI does not install the modules the scripts import.
        if ! pylint --errors-only --disable=import-error "$file" > $result 2>&1; then
          failed=1
          echo "======================================" >> $errlog
          echo "pylint error from $file" >> $errlog
          cat $result >> $errlog
          printf '\n\n' >> $errlog
        fi
      ;;
    esac
  fi
done

echo "::endgroup::"

if [ $failed = 1 ]; then
    echo "::error pylint has found errors in Python files."
    echo "::group::The pylint errors are..."
    cat $errlog
    echo "::endgroup::"
    exit 1
fi
