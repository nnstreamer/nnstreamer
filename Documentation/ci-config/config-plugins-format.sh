#!/usr/bin/env bash

##
# @file config-plugins-format.sh
# @brief add plugin modules for a github repository
#

##################################################################################################################
echo "[MODULE] plugins-good: Plugins with good quality"
# Please append your plugin modules here.

echo "pr-format-indent is starting."
echo "[MODULE] CI/pr-format-indent: Check the code formatting style with GNU indent"
echo "Current path: $(pwd)."
echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/standalone/plugins-good/pr-format-indent.sh"
source ${REFERENCE_REPOSITORY}/ci/standalone/plugins-good/pr-format-indent.sh
pr-format-indent
echo "pr-format-indent is done."

# echo "pr-format-clang is starting."
# echo "[MODULE] CI/pr-format-clang: Check the code formatting style with clang-format"
# echo "Current path: $(pwd)."
# echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/standalone/plugins-good/pr-format-clang.sh"
# source ${REFERENCE_REPOSITORY}/ci/standalone/plugins-good/pr-format-clang.sh
# pr-format-clang
# echo "pr-format-clang is done."

#echo "pr-format-exclusive-io is starting."
#echo "[MODULE] CI/pr-format-exclusive-vio: Check issue #279. VIO commits should not touch non VIO files."
#echo "Current path: $(pwd)."
#echo "[DEBUG] source ${REFERENCE_REPOSITORY}/ci/standalone/plugins-good/pr-format-exclusive-vio.sh"
#source ${REFERENCE_REPOSITORY}/ci/standalone/plugins-good/pr-format-exclusive-vio.sh
#pr-format-exclusive-vio
#echo "pr-format-exclusive-io is done."


##################################################################################################################
echo "[MODULE] plugins-staging: Plugin group that does not have evaluation and aging test enough"
# Please append your plugin modules here.

