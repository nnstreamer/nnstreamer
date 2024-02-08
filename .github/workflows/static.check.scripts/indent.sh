#!/usr/bin/env bash

##
# Imported from TAOS-CI
# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved.
# Modified for Github-Action in 2024.
#
# Argument 1 ($1): the file containing the list of files to be checked

##
# @file indent.sh
# @brief Check the code formatting style with GNU indent. Orignally pr-prebuild-indent.sh
# @see      https://www.gnu.org/software/indent
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

which indent
if [[ $? -ne 0 ]]; then
  echo "::error The indent utility is not found."
  exit 1
fi

echo "::group::Indent check started"

for file in `cat $files`; do
  if [[ `file $file | grep "ASCII text" | wc -l` -gt 0 ]]; then
    case $file in
      *.c|*.cpp)
        indent \
          --braces-on-if-line \
          --case-brace-indentation0 \
          --case-indentation2 \
          --braces-after-struct-decl-line \
          --line-length80 \
          --no-tabs \
          --cuddle-else \
          --dont-line-up-parentheses \
          --continuation-indentation4 \
          --honour-newlines \
          --tab-size8 \
          --indent-level2 \
          --leave-preprocessor-space \
          $file
      ;;
    esac
  fi
done

tmpfile=$(mktemp)
git diff > ${tmpfile}
PATCHFILESIZE=$(stat -c%s ${tmpfile})
if [[ $PATCHFILESIZE -ne 0 ]]; then
  failed=1
fi

echo "::endgroup::"

if [ $failed = 1 ]; then
    echo "::warning There is an indentation style error."
    echo "::group::The indentation style errors are..."
    cat ${tmpfile}
    echo "::endgroup::"
fi
