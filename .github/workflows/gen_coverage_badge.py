#!/usr/bin/env python3
##
# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2024 Samsung Electronics
#
# @file gen_github_badge_svg.py
# @brief A tool for generating a github badge image in svg format
# @author Wook Song <wook16.song@samsung.com>
# @author Gichan Jang <gichan2.jang@samsung.com>
# @note
# usage :  python gen_badge.py {gcov-index-file} {output.svg}
#
import sys
import os
from bs4 import BeautifulSoup
from pybadges import badge

##
# @brief Get a gradient colorcode from green to red of given value (scaled)
# @param[in] val A value to be conveted to a gradient colorcode (i.e., #FFFFFF)
# @param[in] scale A limit for the val, 0 <= val <= scale
def get_code_g_y_r(val, scale):
    """get_code_g_y_r(val, scale) -> str"""
    if val <= 50:
        red = 255
        green = val * (255 / (float(scale) / 2))
    else:
        green = 255 * val / scale
        red = 255 - (val - 50) * (255 / (float(scale) / 2))

    rgb = (int(red), int(green), int(0))

    return '#%02x%02x%02x' % rgb

##
# @brief Generate a github badge svg file representing code coverage
# @param[in] html A concatenated string of the whole contents in index.html that is the result of LCOV
# @param[in] path A file path to save the svg file
def gen_coverage_badge(html, path):
    #parse LCOV html
    soup = BeautifulSoup(html, 'html.parser')
    lines, line_hits, funcs, func_hits = \
        soup.find('table').find_all('td', {'class': 'headerCovTableEntry'})
    line_hits = float(line_hits.text)
    lines = float(lines.text)
    line_coverage = line_hits / lines
    rgb_code = get_code_g_y_r(line_coverage * 100, 100)
    coverage_str = str(format(line_coverage * 100, '.2f')) + '%'
    s = badge(left_text='coverage', right_text=coverage_str, right_color=rgb_code)

    file = open(path, "w")
    file.write(s)
    file.close()


if __name__ == '__main__':
    # argv[1]: [url/file] a path or url of LCOV html to get information for badge generation
    # argv[2]: [file] a file path to save the generated svg file
    if len(sys.argv) < 3:
        exit(1)

    str_html = ''
    if os.path.isfile(sys.argv[1]):
        with open(sys.argv[1], 'r') as f:
            str_html = f.read()
            if not BeautifulSoup(str_html, "html.parser").find():
                exit(1)
    else:
        exit(1)

    path_out_svg=''
    if not os.access(os.path.dirname(sys.argv[2]) or os.getcwd(), os.W_OK):
        exit(1)
    else:
        path_out_svg = os.path.abspath(sys.argv[2])
        if os.path.isdir(path_out_svg) or os.path.islink(path_out_svg):
          exit(1)

    gen_coverage_badge(str_html, path_out_svg)
