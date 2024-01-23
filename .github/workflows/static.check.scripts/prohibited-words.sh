#!/usr/bin/env bash

##
# Imported from TAOS-CI
# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved.
# Modified for Github-Action in 2024.
#
# Argument 1 ($1): the file containing the list of files to be checked
#

##
# @file     prohibited-words.sh
# @brief    Check if there are prohibited words in the text files.
#           Originally, pr-prebuild-prohibited-words.sh
#
# It to check a prohibited word if there are unnecessary words in the source codes.
#
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

check git
check grep
check wc

# Inspect all files that contributor modifed.
target_files=""
for file in `cat $files`; do
  # Skip obsolete folder
  if [[ $file =~ ^obsolete/.* ]]; then
      continue
  fi
  # Skip external folder
  if [[ $file =~ ^external/.* ]]; then
      continue
  fi
  # Handle only text files among files in one commit.
  if [[ `file $file | grep "ASCII text" | wc -l` -gt 0 ]]; then
    case $file in
      # Declare source code files to inspect a prohibited word
      *.c | *.h | *.cpp | *.hpp| *.py | *.sh | *.php | *.md )
        target_files="$target_files $file"
        ;;
    esac
  fi
done

bad_words_sw="grep"
bad_words_list=".github/workflows/static.check.scripts/prohibited-words.txt"
bad_words_rules="--color -n -r -H -f $bad_words_list"
bad_words_log_file=$(mktemp)

# Run a prohibited-words module in case that a PR includes text files.
if [[ -n "${target_files/[ ]*\n/}" ]]; then
  if [[ ! -f $bad_words_list ]]; then
    echo "::error A prohibited word list file, $bad_words_list, doesn't exist."
    exit 1
  fi

  # Step 1: Run this module to filter prohibited words from a text file.
  # (e.g., grep --color -f "$PROHIBITED_WORDS" $filename)
  $bad_words_sw $bad_words_rules $target_files > ${bad_words_log_file}

  # Step 2: Display the execution result for debugging in case of a failure
  cat ${bad_words_log_file}

  # Step 3: Count prohibited words from variable result_content
  result_count=$(cat ${bad_word_log_file} | grep -c '^' )

  # Step 4: change a value of the check result
  if [[ $result_count -gt 0 ]]; then
    echo "::error There are prohibited words in this PR (counted $result_count)."
    exit 1
  else
    echo "There is no prohibited word found in this PR."
  fi
fi
