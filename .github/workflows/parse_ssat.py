#!/usr/bin/python
##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2025 Samsung Electronics
#
# @file    parse_ssat.py
# @brief   Parse SSAT result to xml format

import sys

if len(sys.argv) < 3:
  sys.exit(1)

LOG_PATH="unittest_result/" + sys.argv[1]
XML_PATH="unittest_result/" + sys.argv[2]
TIMESTAMP=sys.argv[3]

log_f = open(LOG_PATH, 'r')
xml_f = open(XML_PATH, 'w')

xml_dic = {}
total_tests = 0
total_fails = 0
test_group = ""
while True:
  line = log_f.readline()
  if not line: break;

  lines = line.split()
  # test group start
  if len(lines) > 5 and lines[5] == "Starts.":
    test_group = lines[4]
    xml_dic[test_group] = {'passed_tests': [], 'failed_tests': []}
    continue

  # test group end
  if len(lines) > 5 and lines[5] == test_group:
    test_group = ""
    continue

  if test_group != "":
  # check passed
    if len(lines) > 3 and lines[2] == "[PASSED]":
      test_name = ""
      for i in range(3, len(lines)):
        test_name += lines[i];
        if i != len(lines) - 1:
          test_name += " "
      xml_dic[test_group]['passed_tests'].append(test_name)
      total_tests += 1

    # check failed
    if len(lines) > 3 and lines[2] == "[FAILED]":
      test_name = ""
      for i in range(3, len(lines)):
        test_name += lines[i];
        test_name += " "
      xml_dic[test_group]['failed_tests'].append(test_name)
      total_tests += 1
      total_fails += 1

log_f.close()

xml_f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
xml_f.write('<testsuites tests="' + str(total_tests) + '" failures="' + str(total_fails) + '" disabled="0" errors="0" timestamp="' + TIMESTAMP + '" time="0" name="ssatAllTests">\n')

for test_group in xml_dic:
  xml_f.write('\t<testsuite name="' + test_group + '" tests="' + str(len(xml_dic[test_group]['passed_tests'])) + '" failures="' + str(len(xml_dic[test_group]['failed_tests'])) + '" disabled="0" errors="0">\n')
  for test_name in xml_dic[test_group]['passed_tests']:
    xml_f.write('\t\t<testcase name="' + test_name + '" status="run" time="0" classname="' + test_group + '" />\n')
  for test_name in xml_dic[test_group]['failed_tests']:
    xml_f.write('\t\t<testcase name="' + test_name + '" status="run" time="0" classname="' + test_group + '" >\n')
    xml_f.write('\t\t\t<failure message="" type=""/>\n')
    xml_f.write('\t\t</testcase>\n')
  xml_f.write('\t</testsuite>\n')
xml_f.write('</testsuites>\n')
xml_f.close()
