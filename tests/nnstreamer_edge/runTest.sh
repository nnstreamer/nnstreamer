#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author Yechan Choi <yechan9.choi@samsung.com>
## @date 20 Jul 2022
## @brief SSAT Test Cases for NNStreamer
##
if [[ "$SSATAPILOADED" != "1" ]]; then
    SILENT=0
    INDEPENDENT=1
    search="ssat-api.sh"
    source $search
    printf "${Blue}Independent Mode${NC}"
fi

# This is compatible with SSAT (https://github.com/myungjoo/SSAT)
testInit $1

# NNStreamer and plugins path for test
PATH_TO_PLUGIN="../../build"

# TODO: edgesink/EdgeSink naming
check_edgesink=$(gst-inspect-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} edgesink | grep EdgeSink)
if [[ ! $check_edgesink ]]; then
    # TODO: edgesink/EdgeSink naming
    # echo 'Cannot find edge sink plugins. Skip tests.'
    testResult 0 "1" " Check Edge Sink Installation"
    report
    exit
fi

report