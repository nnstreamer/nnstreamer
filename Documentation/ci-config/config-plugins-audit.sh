#!/usr/bin/env bash

##
# @file config-plugins-audit.sh
# @brief add plugin modules for a github repository

###### plugins-base ###############################################################################################
echo "[MODULE] plugins-base: Plugin group that follow Apache license with good quality"
# Please append your plugin modules here.

module_name="pr-audit-build-tizen"
echo "[DEBUG] $module_name is started."
echo "[DEBUG] CI/$module_name: Check if Tizen rpm package is successfully generated."
echo "[DEBUG] Current path: $(pwd)."
source ${REFERENCE_REPOSITORY}/ci/standalone/plugins-base/$module_name.sh
#$module_name
echo "[DEBUG] $module_name is done."

module_name="pr-audit-build-ubuntu"
echo "[DEBUG] $module_name is started."
echo "[DEBUG] CI/$module_name: Check if Ubuntu deb package is successfully generated."
echo "[DEBUG] Current path: $(pwd)."
source ${REFERENCE_REPOSITORY}/ci/standalone/plugins-base/$module_name.sh
#$module_name
echo "[DEBUG] $module_name is done."

###### plugins-good ###############################################################################################
echo "[MODULE] plugins-good: Plugin group that follow Apache license with good quality"
# Please append your plugin modules here.






###### plugins-ugly ################################################################################################
echo "[MODULE] plugins-ugly: Plugin group that does not have evaluation and aging test enough"
# Please append your plugin modules here.

module_name="pr-audit-resource"
echo "[DEBUG] $module_name is started."
echo "[DEBUG] CI/$module_name: Check if there are not-installed resource files."
echo "[DEBUG] Current path: $(pwd)."
source ${REFERENCE_REPOSITORY}/ci/standalone/plugins-ugly/$module_name.sh
$module_name
echo "[DEBUG] $module_name is done."

