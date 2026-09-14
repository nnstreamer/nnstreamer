#!/usr/bin/env bash

##
# Imported from TAOS-CI
# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved.
# Modified for Github-Action in 2024.
#
# Argument 1 ($1): the file containing the list of files to be checked
#

##
# @file     shellcheck.sh
# @brief    This module is a static analysis tool for shell scripts such as sh, bash.
#           Originally, pr-prebuild-shellcheck.sh
#
#  It is mainly focused on handling typical beginner and intermediate level syntax errors
#  and pitfalls where the shell just gives a cryptic error message or strange behavior,
#  but it also reports on a few more advanced issues where corner cases can cause delayed
#  failures.
#
# @see      https://www.shellcheck.net/
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
# Check if required commands are installed by server administrator
check cat
check shellcheck
check file
check grep
check wc

shell_syntax_analysis_sw="shellcheck"
shell_syntax_analysis_rules="-s bash -S error"
shell_syntax_check_result=$(mktemp)

# Inspect all files that contributor modifed.
for file in `cat $files`; do
  # Skip obsolete folder
  if [[ $file =~ ^obsolete/.* ]]; then
    continue
  fi
  # Skip external folder
  if [[ $file =~ ^external/.* ]]; then
    continue
  fi
  # Handle only text files in case that there are lots of files in one commit.
  echo "[DEBUG] file name is ($file)."
  if [[ `file $file | grep "text" | wc -l` -gt 0 ]]; then
    case $file in
      # In case of .sh or .bash file
      *.sh | *.bash)
        echo "($file) file is a shell script file."

        $shell_syntax_analysis_sw $shell_syntax_analysis_rules "$file" > ${shell_syntax_check_result} 2>&1
        result=$?

        echo "::group::shellcheck result of $file"
        cat ${shell_syntax_check_result}
        echo "::endgroup::"

        if [[ $result -ne 0 ]]; then
          echo "$shell_syntax_analysis_sw: failed. file name: $file"
          failed=1
        else
          echo "$shell_syntax_analysis_sw: passed. file name: $file"
        fi
        ;;
    esac
  fi
done

if [[ "$failed" == "1" ]]; then
  echo "::error shellcheck has found errors in shell scripts. Please refer to the log above."
  exit 1
fi
