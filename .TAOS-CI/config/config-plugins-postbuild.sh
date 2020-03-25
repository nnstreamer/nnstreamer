#!/usr/bin/env bash

# Do not append a license statement in the configuration file for a differnet license-based repository.

##
# @file     config-plugins-postbuild.sh
# @brief    Configuraiton file to maintain postbuild modules (after completing a build procedure)
# @see      https://github.com/nnsuite/TAOS-CI
# @author   Geunsik Lim <geunsik.lim@samsung.com>

##### Set environment for plug-in check modules of the postbuild group
declare -i idx=-1

###### plugins-base ###############################################################################################
echo "[MODULE] plugins-base: Plugin group is a well-maintained collection of plugin modules."
# Please append your plugin modules here.

postbuild_plugins[++idx]="pr-postbuild-build-tizen"
echo "[DEBUG] The default BUILD_MODE of ${postbuild_plugins[idx]} is declared with 99 (SKIP MODE) by default in plugins-base folder."
echo "[DEBUG] ${postbuild_plugins[idx]} is started."
echo "[DEBUG] ${BOT_NAME}/${postbuild_plugins[idx]}: Check if Tizen rpm package is successfully generated."
echo "[DEBUG] Current path: $(pwd)."
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-base/${postbuild_plugins[idx]}.sh


postbuild_plugins[++idx]="pr-postbuild-build-ubuntu"
echo "[DEBUG] The default BUILD_MODE of ${postbuild_plugins[idx]} is declared with 99 (SKIP MODE) by default in plugins-base folder."
echo "[DEBUG] ${postbuild_plugins[idx]} is started."
echo "[DEBUG] ${BOT_NAME}/${postbuild_plugins[idx]}: Check if Ubuntu deb package is successfully generated."
echo "[DEBUG] Current path: $(pwd)."
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-base/${postbuild_plugins[idx]}.sh


postbuild_plugins[++idx]="pr-postbuild-build-yocto"
echo "[DEBUG] The default BUILD_MODE of ${postbuild_plugins[idx]} is declared with 99 (SKIP MODE) by default in plugins-base folder."
echo "[DEBUG] ${postbuild_plugins[idx]} is started."
echo "[DEBUG] ${BOT_NAME}/${postbuild_plugins[idx]}: Check if YOCTO deb package is successfully generated."
echo "[DEBUG] Current path: $(pwd)."
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-base/${postbuild_plugins[idx]}.sh


postbuild_plugins[++idx]="pr-postbuild-build-android"
echo "[DEBUG] The default BUILD_MODE of ${postbuild_plugins[idx]} is declared with 99 (SKIP MODE) by default in plugins-base folder."
echo "[DEBUG] ${postbuild_plugins[idx]} is started."
echo "[DEBUG] ${BOT_NAME}/${postbuild_plugins[idx]}: Check if Android package is successfully generated."
echo "[DEBUG] Current path: $(pwd)."
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-base/${postbuild_plugins[idx]}.sh

###### plugins-good ###############################################################################################
echo "[MODULE] plugins-good: Plugin group that follow Apache license with good quality"
# Please append your plugin modules here.



###### plugins-staging ################################################################################################
echo "[MODULE] plugins-staging: Plugin group that does not have evaluation and aging test enough"
# Please append your plugin modules here.

# postbuild_plugins[++idx]="pr-postbuild-resource"
# echo "[DEBUG] ${postbuild_plugins[idx]} is started."
# echo "[DEBUG] ${BOT_NAME}/${postbuild_plugins[idx]}: Check if there are not-installed resource files."
# echo "[DEBUG] Current path: $(pwd)."
# source ${REFERENCE_REPOSITORY}/ci/taos/plugins-staging/${postbuild_plugins[idx]}.sh


postbuild_plugins[++idx]="pr-postbuild-nnstreamer-ubuntu-apptest"
echo "[DEBUG] ${postbuild_plugins[idx]} is started."
echo "[DEBUG] ${BOT_NAME}/${postbuild_plugins[idx]}: Check nnstreamer sample app"
echo "[DEBUG] Current path: $(pwd)."
source ${REFERENCE_REPOSITORY}/ci/taos/plugins-staging/${postbuild_plugins[idx]}.sh


