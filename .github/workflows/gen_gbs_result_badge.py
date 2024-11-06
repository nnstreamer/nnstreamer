#!/usr/bin/env python3
##
# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2024 Samsung Electronics
#
# @file gen_gbs_result_badge.py
# @brief A tool for generating a github badge image in svg format
# @author Gichan Jang <gichan2.jang@samsung.com>
#
import sys
import os
import json
from pybadges import badge

##
# @brief Save the badge on the given path
# @param[in] path A file path to save the svg file
# @param[in] badge github badge svg file
def save_badge (path, badge):
    try:
        with open(path, "w") as f:
            f.write(badge)
    except IOError as e:
        print("Failed to save badge:", str(e))

##
# @brief Generate a github badge svg file representing daily build result
# @param[in] html build result of arch.
# @param[in] path A directory path to save the svg file
def gen_build_result_badge(json_str, out_dir):
    daily_result="success"
    for arch in json_str:
        try:
            result = json_str[arch]
            daily_color = color= 'green'
            if result == 'failure':
              color = 'red'
              daily_result = result
              daily_color = 'red'

            s = badge(left_text='test', right_text=result, right_color=color)
            path = os.path.join(out_dir, arch + '_reuslt.svg')
            save_badge (path, s)
        except Exception as e:
            print(f"An error occurred while processing {arch} :", str(e))
    try:
        s = badge(left_text='Daily build', right_text=result, right_color=color)
        path = os.path.join(out_dir, 'daily_reuslt.svg')
        save_badge (path, s)
    except Exception as e:
        print("An error occurred while processing daily build badge:", str(e))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        exit(1)

    try:
        with open(sys.argv[1], "r") as input_file:
            data = json.load(input_file)
            gen_build_result_badge(data, sys.argv[2])
    except Exception as e:
        print("An error occurred:", str(e))
