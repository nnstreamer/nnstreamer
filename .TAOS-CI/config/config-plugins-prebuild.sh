#!/usr/bin/env bash

# Do not append a license statement in the configuration file for a differnet license-based repository.

##
# @file     config-plugins-prebuild.sh
# @brief    Configuration file to maintain modules of prebuild group (before doing a build procedure)
# @see      https://github.com/nnsuite/TAOS-CI
# @author   Geunsik Lim <geunsik.lim@samsung.com>


##### Set environment for plugin modules of prebuild group
declare -i idx=-1

##################################################################################################################
echo "[MODULE] plugins-good: Plugin group that follow Apache license with good quality"
# Please append your plugin modules here.

prebuild_plugins[++idx]="pr-prebuild-doxygen-tag"
echo "${prebuild_plugins[idx]} is starting."
echo "[MODULE] ${BOT_NAME}/${prebuild_plugins[idx]}: Check a source code consists of required doxygen tags."
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh

prebuild_plugins[++idx]="pr-prebuild-indent"
echo "${prebuild_plugins[idx]} is starting."
echo "[MODULE] ${BOT_NAME}/${prebuild_plugins[idx]}: Check the code formatting style with GNU indent"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh


#prebuild_plugins[++idx]="pr-prebuild-clang"
# echo "${prebuild_plugins[idx]} is starting."
# echo "[MODULE] ${BOT_NAME}/${prebuild_plugins[idx]}: Check the code formatting style with clang-format"
# echo "[DEBUG] The current path: $(pwd)."
# echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh"
# source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh

#prebuild_plugins[++idx]="pr-prebuild-exclusive-vio"
# echo "${prebuild_plugins[idx]} is starting."
# echo "[MODULE] ${BOT_NAME}/${prebuild_plugins[idx]}: Check issue #279. VIO commits should not touch non VIO files."
# echo "[DEBUG] The current path: $(pwd)."
# echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh"
# source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh

prebuild_plugins[++idx]="pr-prebuild-pylint"
echo "${prebuild_plugins[idx]} is starting."
echo "[MODULE] ${BOT_NAME}/${prebuild_plugins[idx]}: Check the code formatting style with pylint"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh

prebuild_plugins[++idx]="pr-prebuild-newline"
echo "${prebuild_plugins[idx]} is starting."
echo "[MODULE] ${BOT_NAME}/${prebuild_plugins[idx]}: Check if there is a newline issue in text files"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh

prebuild_plugins[++idx]="pr-prebuild-rpm-spec"
echo "${prebuild_plugins[idx]} is starting."
echo "[MODULE] ${BOT_NAME}/${prebuild_plugins[idx]}: Check if there is incorrect staements in *.spec file"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh

prebuild_plugins[++idx]="pr-prebuild-file-size"
echo "${prebuild_plugins[idx]} is starting."
echo "[MODULE] ${BOT_NAME}/${prebuild_plugins[idx]}: Check the file size to not include big binary files"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh

prebuild_plugins[++idx]="pr-prebuild-cppcheck"
echo "${prebuild_plugins[idx]} is starting."
echo "[MODULE] ${BOT_NAME}/${prebuild_plugins[idx]}: Check dangerous coding constructs in source codes (*.c, *.cpp) with cppcheck"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh

prebuild_plugins[++idx]="pr-prebuild-nobody"
echo "${prebuild_plugins[idx]} is starting."
echo "[MODULE] ${BOT_NAME}/${prebuild_plugins[idx]}: Check the commit message body"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh

prebuild_plugins[++idx]="pr-prebuild-timestamp"
echo "${prebuild_plugins[idx]} is starting."
echo "[MODULE] ${BOT_NAME}/${prebuild_plugins[idx]}: Check the timestamp of the commit"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh

prebuild_plugins[++idx]="pr-prebuild-executable"
echo "${prebuild_plugins[idx]} is starting."
echo "[MODULE] ${BOT_NAME}/${prebuild_plugins[idx]}:  Check executable bits for .cpp, .c, .hpp, .h, .prototxt, .caffemodel, .txt., .init"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh

prebuild_plugins[++idx]="pr-prebuild-hardcoded-path"
echo "${prebuild_plugins[idx]} is starting."
echo "[MODULE] ${BOT_NAME}/${prebuild_plugins[idx]}: Check prohibited hardcoded paths (/home/* for now)"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh

prebuild_plugins[++idx]="pr-prebuild-misspelling"
echo "${prebuild_plugins[idx]} is starting."
echo "[MODULE] ${BOT_NAME}/${prebuild_plugins[idx]}: Check a misspelled statement in a document file with GNU Aspell"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh

prebuild_plugins[++idx]="pr-prebuild-doxygen-build"
echo "${prebuild_plugins[idx]} is starting."
echo "[MODULE] ${BOT_NAME}/${prebuild_plugins[idx]}: Check a doxygen grammar if a doxygen can normally generates source code"

prebuild_plugins[++idx]="pr-prebuild-sloccount"
echo "${prebuild_plugins[idx]} is starting."
echo "[MODULE] ${BOT_NAME}/${prebuild_plugins[idx]}: Check physical Source Lines of Code (SLOC) in a source code"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh

prebuild_plugins[++idx]="pr-prebuild-prohibited-words"
echo "${prebuild_plugins[idx]} is starting."
echo "[MODULE] ${BOT_NAME}/${prebuild_plugins[idx]}: Check if source codes have prohibited words."
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh

prebuild_plugins[++idx]="pr-prebuild-signed-off-by"
echo "${prebuild_plugins[idx]} is starting."
echo "[MODULE] ${BOT_NAME}/${prebuild_plugins[idx]}: Check 'Signed-off-by' in commit body"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh

prebuild_plugins[++idx]="pr-prebuild-shellcheck"
echo "${prebuild_plugins[idx]} is starting."
echo "[MODULE] ${BOT_NAME}/${prebuild_plugins[idx]}: Check a syntax error in a shell script file with 'shellcheck' package"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh

prebuild_plugins[++idx]="pr-prebuild-flawfinder"
echo "${prebuild_plugins[idx]} is starting."
echo "[MODULE] ${BOT_NAME}/${prebuild_plugins[idx]}: Check security problems in the C/C++ source code with 'flawfinder' package"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh

prebuild_plugins[++idx]="pr-prebuild-coverity"
echo "${prebuild_plugins[idx]} is starting."
echo "[MODULE] ${BOT_NAME}/${prebuild_plugins[idx]}: Check defects in the C/C++ source code with 'coverity' package"
echo "[DEBUG] The current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh"
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-good/${prebuild_plugins[idx]}.sh

##################################################################################################################
echo "[MODULE] plugins-staging: Plugin group that does not have an evaluation and aging test enough"
# Please append your plugin modules here.

