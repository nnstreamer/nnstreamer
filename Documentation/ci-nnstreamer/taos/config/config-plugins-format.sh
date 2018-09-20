#!/usr/bin/env bash

##
# @file     config-plugins-format.sh
# @brief    Configuration file to maintain format modules (before doing a build procedure)
# @see      https://github.com/nnsuite/TAOS-CI
# @author   Geunsik Lim <geunsik.lim@samsung.com>


##### Set environment for format plugins
declare -i idx=-1

##################################################################################################################
echo "[MODULE] plugins-good: Plugin group that follow a writing rule with good quality"
# Please append your plugin modules here.

format_plugins[++idx]="pr-format-doxygen-tag"
echo "${format_plugins[idx]} is starting."
echo "[MODULE] TAOS/${format_plugins[idx]}: Check a source code consists of required doxygen tags."
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh

format_plugins[++idx]="pr-format-indent"
echo "${format_plugins[idx]} is starting."
echo "[MODULE] TAOS/${format_plugins[idx]}: Check the code formatting style with GNU indent"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh


#format_plugins[++idx]="pr-format-clang"
# echo "${format_plugins[idx]} is starting."
# echo "[MODULE] TAOS/${format_plugins[idx]}: Check the code formatting style with clang-format"
# echo "[DEBUG] The current path: $(pwd)."
# echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh"
# source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh

#format_plugins[++idx]="pr-format-exclusive-vio"
# echo "${format_plugins[idx]} is starting."
# echo "[MODULE] TAOS/${format_plugins[idx]}: Check issue #279. VIO commits should not touch non VIO files."
# echo "[DEBUG] The current path: $(pwd)."
# echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh"
# source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh

format_plugins[++idx]="pr-format-pylint"
echo "${format_plugins[idx]} is starting."
echo "[MODULE] TAOS/${format_plugins[idx]}: Check the code formatting style with pylint"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh

format_plugins[++idx]="pr-format-newline"
echo "${format_plugins[idx]} is starting."
echo "[MODULE] TAOS/${format_plugins[idx]}: Check if there is a newline issue in text files"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh

format_plugins[++idx]="pr-format-rpm-spec"
echo "${format_plugins[idx]} is starting."
echo "[MODULE] TAOS/${format_plugins[idx]}: Check if there is incorrect staements in *.spec file"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh

format_plugins[++idx]="pr-format-file-size"
echo "${format_plugins[idx]} is starting."
echo "[MODULE] TAOS/${format_plugins[idx]}: Check the file size to not include big binary files"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh

format_plugins[++idx]="pr-format-cppcheck"
echo "${format_plugins[idx]} is starting."
echo "[MODULE] TAOS/${format_plugins[idx]}: Check dangerous coding constructs in source codes (*.c, *.cpp) with cppcheck"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh

format_plugins[++idx]="pr-format-nobody"
echo "${format_plugins[idx]} is starting."
echo "[MODULE] TAOS/${format_plugins[idx]}: Check the commit message body"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh

format_plugins[++idx]="pr-format-timestamp"
echo "${format_plugins[idx]} is starting."
echo "[MODULE] TAOS/${format_plugins[idx]}: Check the timestamp of the commit"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh

format_plugins[++idx]="pr-format-executable"
echo "${format_plugins[idx]} is starting."
echo "[MODULE] TAOS/${format_plugins[idx]}:  Check executable bits for .cpp, .c, .hpp, .h, .prototxt, .caffemodel, .txt., .init"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh

format_plugins[++idx]="pr-format-hardcoded-path"
echo "${format_plugins[idx]} is starting."
echo "[MODULE] TAOS/${format_plugins[idx]}: Check prohibited hardcoded paths (/home/* for now)"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh

format_plugins[++idx]="pr-format-misspelling"
echo "${format_plugins[idx]} is starting."
echo "[MODULE] TAOS/${format_plugins[idx]}: Check a misspelled statement in a document file with GNU Aspell"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh

format_plugins[++idx]="pr-format-doxygen-build"
echo "${format_plugins[idx]} is starting."
echo "[MODULE] TAOS/${format_plugins[idx]}: Check a doxygen grammar if a doxygen can normally generates source code"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${format_plugins[idx]}.sh


##################################################################################################################
echo "[MODULE] plugins-staging: Plugin group that does not have an evaluation and aging test enough"
# Please append your plugin modules here.

