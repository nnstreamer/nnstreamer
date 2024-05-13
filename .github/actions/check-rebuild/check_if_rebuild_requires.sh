#!/usr/bin/env bash

##
# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file: check_if_rebuild_requires.sh
# @brief    Check if rebuild & unit-test is required with the given PR.
# @see      https://github.com/nnstreamer/nnstreamer
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>
#
# Argument 1 ($1): the file containing list of files to be checked.
# Argument 2 ($2): build mode to be checked
#                  gbs: check if Tizen GBS build is required
#                  debian: check if pdebuild is required
#                  android: check if jni rebuild is required
#                  build (default): check if general meson rebuild is required.

if [ -z $1 ]; then
  echo "::error The argument (file path) is not given."
  exit 1
fi

if [ -z $2 ]; then
  mode="build"
else
  mode=$2
fi

rebuild=0
regbs=0
redebian=0
reandroid=0

for file in `cat $1`; do
  case $file in
    *.md|*.png|*.webp|*.css|*.html )
    ;;
    packaging/* )
      regbs='1'
      ;;
    debian/* )
      redebian='1'
      ;;
    jni/* )
      reandroid='1'
      ;;
    * )
      rebuild='1'
      regbs='1'
      redebian='1'
      reandroid='1'
      ;;
  esac
done

case $mode in
  gbs)
    if [[ "$regbs" == "1" ]]; then
      echo "REBUILD=YES"
      exit 0
    fi
    ;;
  debian)
    if [[ "$redebian" == "1" ]]; then
      echo "REBUILD=YES"
      exit 0
    fi
    ;;
  android)
    if [[ "$reandroid" == "1" ]]; then
      echo "REBUILD=YES"
      exit 0
    fi
    ;;
  *)
    if [[ "$rebuild" == "1" ]]; then
      echo "REBUILD=YES"
      exit 0
    fi
    ;;
esac

echo "REBUILD=NO"
