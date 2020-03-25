#!/usr/bin/env bash

# Do not append a license statement in the configuration file for a differnet license-based repository.

##
# @file config-server-administrator.sh
# @brief configuration file to declare contents that a server administrator installed.
# @see      https://github.com/nnsuite/TAOS-CI
# @author   Geunsik Lim <geunsik.lim@samsung.com>
#

########### Caution: If you are not server administrator, do not modify this file #################

# Note that administrator of a server has to specify the location of eSDK at first.
# In order to know how to install eSDK, please read plugins-base/pr-postbuild-build-yocto.sh file.
# It is environment variables that are imported from eSDK to use devtool command.
# - YOCTO_ESDK_DIR="/var/www"
# - YOCTO_ESDK_NAME="kairos_sdk" or YOCTO_ESDK_NAME="poky_sdk"
# In general, root path of Yocto eSDK is declated in $YOCTO_ESDK_DIR/$YOCTO_ESDK_NAME/ folder.

YOCTO_ESDK_DIR="/var/www/"
YOCTO_ESDK_NAME=""

