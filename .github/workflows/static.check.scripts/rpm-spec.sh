#!/usr/bin/env bash

##
# Imported from TAOS-CI
# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved.
# Modified for Github-Action in 2024.
#
# Argument 1 ($1): the file containing the list of files to be checked
# Argument 2 ($2): the max number of errors to be tolerated

##
# @file     rpm-spec.sh
# @brief    Check the spec file (packaging/*.spec) with rpmlint. Originally pr-prebuild-rpm-spec.sh
# @see      https://github.com/nnstreamer/TAOS-CI
# @see      https://github.com/nnstreamer/nnstreamer
# @author   Geunsik Lim <geunsik.lim@samsung.com>
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>
#

if [ -z $1 ]; then
  echo "::error The argument (file path) is not given."
  exit 1
fi

tolerable=0
if [ -z $2 ]; then
  echo "Tolarable rpmlint errors = 0"
else
  tolerable=$2
  echo "Tolarable rpmlint errors = $2"
fi

files=$1
failed=0

if [ ! -f $files ]; then
  echo "::error The file $files does not exists."
  exit 1
fi

spec_modified="false"
specfiles=""
resultfile=$(mktemp)

for file in `cat $files`; do
  if [[ $file =~ ^obsolete/.* ]]; then
      continue
  fi
  if [[ $file =~ ^external/.* ]]; then
      continue
  fi
  # Handle only spec file in case that there are lots of files in one commit.
  if [[ `file $file | grep "ASCII text" | wc -l` -gt 0 ]]; then
    case $file in
      *.spec)
        specfiles+=" ${file}"
        echo "A .spec file found: $file"
        spec_modified="true"
        break
      ;;
    esac
  fi
done

if [[ spec_modified == "true" ]]; then
  rpmlint $specfiles | aha --line-fix > $resultfile
  echo "::group::rpmlint result"
  cat $resultfile
  echo "::ungroup::"

  count=`grep "[0-9]* errors, [0-9]* warnings." $resultfile | grep -o "[0-9]* errors" | grep -o "[0-9]*"`
  if [ $count -gt $tolerable ]; then
    echo "::error RPMLINT reports more errors ($count) than tolerated ($tolerable)."
    exit 1
  fi
else
  echo "There is no .spec file modified in this PR. RPMLINT is not executed."
fi
