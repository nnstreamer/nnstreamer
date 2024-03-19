#!/usr/bin/env bash

##
# Imported from TAOS-CI
# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved.
# Modified for Github-Action in 2024.
#
# Argument 1 ($1): the file containing the list of files to be checked
#

##
# @file     misspelling.sh
# @brief    Check a misspelled statement in a text document file with GNU Aspell
#           Originally, pr-prebuild-misspelling.sh
#
# GNU Aspell is a Free and Open Source spell checker designed to eventually replace Ispell.
# It can either be used as a library or as an independent spell checker. Its main feature
# is that it does a superior job of suggesting possible replacements for a misspelled word
# than just about any other spell checker out there for the English language. Unlike Ispell,
# Aspell can also easily check documents in UTF-8 without having to use a special dictionary.
#
# @see      http://aspell.net/
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
check aspell
check file
check cat

errorreport=$(mktemp)

for file in `cat $files`; do
  # skip obsolete folder
  if [[ $file =~ ^obsolete/.* ]]; then
      continue
  fi
  # skip external folder
  if [[ $file =~ ^external/.* ]]; then
      continue
  fi

  if [[ `file $file | grep "ASCII text" | wc -l` -gt 0 ]]; then
    # in case of document file: *.md, *.txt)
    case $file in
      # in case of MarkDown(MD) and text file
      *.md | *.txt)
        echo "Checking $file"
        typo_analysis_sw="aspell"
        typo_analysis_rules=" list -l en "
        typo_check_result=$(mktemp)

        cat $file | $typo_analysis_sw $typo_analysis_rules > ${typo_check_result}

        line_count=`cat ${typo_check_result} | wc -l`
        # 9,000 is declared by heuristic method from our experiment.
        if  [[ $line_count -gt 9000 ]]; then
          echo "$typo_analysis_sw: failed. file name: $file, There are $line_count typo(s)."
          error=1
        fi
        ;;
    esac
  fi
done

if [[ "$error" == "1" ]]; then
  echo "::error There are typo error reported by $typo_analysis_sw."
  exit 1
fi
