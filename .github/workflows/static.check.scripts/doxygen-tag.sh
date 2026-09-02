#!/usr/bin/env bash

##
# Imported from TAOS-CI
# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved.
# Modified for Github-Action in 2024.
#
# Argument 1 ($1): the file containing the list of files to be checked

##
# @file doxygen-tag.sh
# @brief Check if there is a doxygen-tag issue. Originally pr-prebuild-doxygen-tag.sh
# @see      https://github.com/nnstreamer/TAOS-CI
# @see      https://github.com/nnstreamer/nnstreamer
# @author   Geunsik Lim <geunsik.lim@samsung.com>
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>
#

if [ -z $1 ]; then
  echo "::error The argument (file path) is not given."
  exit 1
fi

advanced=0
if [ "$2" = "1" ]; then
  advanced=1
fi

files=$1
failed=0
report_path=${report_path:-/dev/null}

if [ ! -f $files ]; then
  echo "::error The file $files does not exists."
  exit 1
fi

##
# @brief Tell whether the declaration starting at a line carries a trailing
#        "/**<" Doxygen comment, on the declaration itself or right after it.
# @param $1 1-based line number where ctags placed the function
# @return 0 when such a comment is found, 1 otherwise
# Reads the global array file_lines (0-based). A declaration ends at the first
# ';' outside braces, or at the '}' that closes an inline body; the comment may
# follow on the next line.
has_trailing_doc() {
  local -i i=$1-1
  local -i last=${#file_lines[@]}-1
  local -i depth=0
  local l opens closes
  while [[ $i -le $last ]]; do
    l=${file_lines[$i]}
    [[ $l == *"/**<"* ]] && return 0
    opens=${l//[^\{]/}
    closes=${l//[^\}]/}
    depth+=${#opens}-${#closes}
    if [[ $depth -le 0 && ( $l == *";"* || $l == *"}"* ) ]]; then
      [[ $i -lt $last && ${file_lines[$i+1]} == *"/**<"* ]] && return 0
      return 1
    fi
    i+=1
  done
  return 1
}

echo "::group::Doxygen tag check started"

for file in `cat $files`; do
  if [[ `file $file | grep "ASCII text" | wc -l` -gt 0 ]]; then
      # In case of source code files: *.c|*.h|*.cpp|*.py|*.sh|*.php )
      case $file in
          # In case of C/C++ code
          *.c|*.h|*.cc|*.hh|*.cpp|*.hpp )
              echo "[DEBUG] ( $file ) file is source code with the text format." >> $report_path
              doxygen_lang="doxygen-cncpp"
              # Append a doxgen rule step by step
              doxygen_basic_rules="@file @brief" # @file and @brief to inspect file
              doxygen_advanced_rules="@author @bug" # @author, @bug to inspect file, @brief for to inspect function

              # Apply advanced doxygen rule if pr_doxygen_check_level=1 in config-environment.sh
              if [[ $advanced == 1 ]]; then
                  doxygen_basic_rules="$doxygen_basic_rules $doxygen_advanced_rules"
              fi

              for word in $doxygen_basic_rules
              do
                  doxygen_rule_compare_count=`cat ${file} | grep "$word" | wc -l`
                  doxygen_rule_expect_count=1

                  # Doxygen_rule_compare_count: real number of Doxygen tags in a file
                  # Doxygen_rule_expect_count: required number of Doxygen tags
                  if [[ $doxygen_rule_compare_count -lt $doxygen_rule_expect_count ]]; then
                      echo "[ERROR] $doxygen_lang: failed. file name: $file, $word tag is required at the top of file"
                      failed=1
                  fi
              done

              # Checking tags for each function
              if [[ $advanced == 1 ]]; then
                  declare -i idx=0
                  brief=0
                  function_positions="" # Line number of functions.
                  structure_positions="" # Line number of structure.
                  mapfile -t file_lines < "$file"

                  # Definitions are checked everywhere. Prototypes are checked
                  # only in headers, where the declaration is the documented
                  # form; a forward declaration in a .c/.cc is documented at
                  # its definition.
                  case $file in
                      *.h|*.hh|*.hpp ) function_check_flag="f+p" ;;
                      * ) function_check_flag="f" ;;
                  esac

                  # Find line number of functions using ctags, and append them.
                  while IFS='' read -r line || [[ -n "$line" ]]; do
                      temp=`echo $line | cut -d ' ' -f3` # line number of function place 3rd field when divided into ' ' >> $report_path
                      function_positions="$function_positions $temp "
                  done < <(ctags -x --c-kinds=$function_check_flag $file) # "--c-kinds=f" mean find function

                  # Find line number of structure using ctags, and append them.
                  while IFS='' read -r line || [[ -n "$line" ]]; do
                      temp=`echo $line | cut -d ' ' -f3` # line number of structure place 3rd field when divided into ' ' >> $report_path
                      structure_positions="$structure_positions $temp "
                  done < <(ctags -x --c-kinds=sc $file) # "--c-kinds=sc" mean find 's'truct and 'c'lass

                  # Checking committed file line by line for detailed hints when missing Doxygen tags.
                  while IFS='' read -r line || [[ -n "$line" ]]; do
                      idx+=1

                      # Check if a function has @brief tag or not.
                      # To pass correct line number not sub number, keep space " $idx ".
                      # ex) want to pass 143 not 14, 43, 1, 3, 4
                      if [[ $function_positions =~ " $idx " && $brief -eq 0 ]] && ! has_trailing_doc $idx; then
                          echo "[ERROR] File name: $file, $idx line, `echo $line | cut -d ' ' -f1` function needs @brief tag "
                          failed=1
                      fi

                      # Check if a structure has @brief tag or not.
                      # To pass correct line number not sub number, keep space " $idx ".
                      # For example, we want to pass 143 not 14, 43, 1, 3, and 4.
                      if [[ $structure_positions =~ " $idx " && $brief -eq 0 ]]; then # same as above.
                          echo "[ERROR] File name: $file, $idx line, structure needs @brief tag "
                          failed=1
                      fi

                      # Find brief or copydoc tag in the comments between the codes.
                      if [[ $line =~  "@brief" || $line =~ "@copydoc" ]]; then
                          brief=1
                      # Doxygen tags become zero in code section.
                      elif [[ $line != *"*"*  && ( $line =~ ";" || $line =~ "}" || $line =~ "#") ]]; then
                          brief=0
                      fi

                      # Check a comment statement that begins with '/*'.
                      # Note that doxygen does not recognize a comment  statement that start with '/*'.
                      # Let's skip the doxygen tag inspection such as "/**" in case of a single line comment.
                      if [[ $line =~ "/*" && $line != *"/**"*  && ( $line != *"*/"  || $line =~ "@" ) && ( $idx != 1 ) ]]; then
                          echo "[ERROR] File name: $file, $idx line, Doxygen or multi line comments should begin with /**"
                          failed=1
                      fi

                      # Check the doxygen tag written in upper case beacuase doxygen cannot use upper case tag such as '@TODO'.
                      # Let's check a comment statement that begins with '//','/*' or ' *'.
                      if  [[ ($line =~ "@"[A-Z]) && ($line == *([[:blank:]])"/*"* || $line == *([[:blank:]])"//"* || $line == *([[:blank:]])"*"*) ]]; then
                          echo "[ERROR] File name: $file, $idx line, The doxygen tag sholud be written in lower case."
                          failed=1
                      fi

                  done < "$file"
              fi
              ;;
          # In case of Python code
          *.py )
              doxygen_lang="doxygen-python"
              # Append a Doxgen rule step by step
              doxygen_rules="@package @brief"
              doxygen_rule_num=0
              doxygen_rule_all=0

              for word in $doxygen_rules
              do
                  doxygen_rule_all=$(( doxygen_rule_all + 1 ))
                  doxygen_rule[$doxygen_rule_all]=`cat ${file} | grep "$word" | wc -l`
                  doxygen_rule_num=$(( $doxygen_rule_num + ${doxygen_rule[$doxygen_rule_all]} ))
              done
              if  [[ $doxygen_rule_num -le 0 ]]; then
                  echo "[ERROR] $doxygen_lang: failed. file name: $file, ($doxygen_rule_num)/$doxygen_rule_all tags are found."
                  failed=1
                  break
              else
                  echo "[DEBUG] $doxygen_lang: passed. file name: $file, ($doxygen_rule_num)/$doxygen_rule_all tags are found." >> $report_path
              fi
              ;;
      esac
  fi
done

echo "::endgroup::"

if [ $failed = 1 ]; then
  echo "::error There is a doxygen tag missing or incorrect."
  exit 1
fi
