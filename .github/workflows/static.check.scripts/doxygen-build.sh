#!/usr/bin/env bash

##
# Imported from TAOS-CI
# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved.
# Modified for Github-Action in 2024.
#
# Argument 1 ($1): the file containing the list of files to be checked
#

##
# @file     doxygen-build.sh
# @brief    Check a doxygen grammar if a doxygen can normally generates source code
#           Originally, pr-prebuild-doxygen-build.sh
#
# Doxygen is the de facto standard tool for generating documentation from annotated C++
# sources, but it also supports other popular programming languages such as C, Objective-C,
# C#, PHP, Java, Python, IDL (Corba, Microsoft, and UNO/OpenOffice flavors), Fortran, VHDL,
# Tcl, and to some extent D.
#
# @see      http://www.doxygen.nl/
# @see      https://github.com/nnstreamer/TAOS-CI
# @see      https://github.com/nnstreamer/nnstreamer
# @author   Geunsik Lim <geunsik.lim@samsung.com>
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>
# @note
#  Note that module developer has to execute a self evaluaton if the plug-in module includes incorrect grammar(s).
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
check file
check grep
check cat
check wc
check doxygen

errorresult=$(mktemp)
doxygen_check_result="doxygen_build_result.txt"

# Inspect all files that contributor modifed.
for file in `cat $files`; do
  # skip obsolete folder
  if [[ $file =~ ^obsolete/.* ]]; then
      continue
  fi
  # skip external folder
  if [[ $file =~ ^external/.* ]]; then
      continue
  fi

  # Handle only a source code sequentially in case that there are lots of files in one commit.
  if [[ `file $file | grep "ASCII text" | wc -l` -gt 0 ]]; then
    case $file in
      # In case of source code
      *.c | *.cpp | *.cc | *.hh | *.h | *.hpp | *.py | *.sh | *.php | *.java)
        doxygen_analysis_sw="doxygen"
        doxygen_analysis_rules=" - "
        doxygen_analysis_config=".github/workflows/static.check.scripts/Doxyfile.prj"

        # Doxygen Usage: ( cat ../Doxyfile.ci ; echo "INPUT=./webhook.php" ) | doxygen -
        ( cat $doxygen_analysis_config ; echo "INPUT=$file" ) | $doxygen_analysis_sw $doxygen_analysis_rules
        result=$?

        if  [[ $result != 0 ]]; then
          failed=1
          echo "====================================================" >> $errorresult
          echo "Doxygen has failed in file $file" >> $errorresult
          echo "====================================================" >> $errorresult
          cat $doxygen_check_result >> $errorresult
          echo "====================================================\n\n" >> $errorresult
        fi
        ;;
    esac
  fi
done

if [[ $failed == "1" ]]; then
  echo "::group::There are doxygen build errors in the pull-requset"
  cat $errorresult
  echo "::endgroup::"
  echo "::error Doxygen build test has failed."
  exit 1
fi
