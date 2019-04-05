#!/usr/bin/env python2.7

##
# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

##
# @file unittestcoverage.py
# @brief Calculate and show unit test coverate rate.
# @author MyungJoo Ham <myungjoo.ham@samsung.com>
# @note
# Precondition:
#  The user must have executed cmake/make build for all compoennts with -fprofile-arcs -ftest-coverage enabled
#  All the unit tests binaries should have been executed.
#  Other than the unit test binaries, no other built binaries should be executed, yet
#
# Usage: (for the case of STAR/AuDri.git)
#
#  $ unittestcoverage module /home/abuild/rpmbuild/BUILD/audri-1.1.1/ROS/autodrive/
#  Please use absolute path to the module directory.
#
#  $ unittestcoverage all /home/abuild/rpmbuild/BUILD/audri-1.1.1/ROS/
#  Please use absolute path to the ROS module root dir
#
# Limitation of this version: supports c/c++ only (.c, .cpp, .h, .hpp)
#

from __future__ import print_function
import re
import os
import os.path
import sys

debugprint = 0

## @brief Debug Print
#
# @param str The string to be debug-printed
def dprint(str):
  global debugprint
  if debugprint == 1:
    print(str)

## @brief Search for c/c++ files not being detected by gcov
#
# @param gcovOutput output of gcov
# @param path Path to be audited
def auditEvaders(gcovOutput, path):
  out = gcovOutput

  targetFiles = {}
  # Generate target file lists
  dprint("Walking in " + path)
  for root, dirs, files in os.walk(path):
    for file in files:
      # TODO 1 : Support other than C/C++
      # TODO 2 : case insensitive
      if file.endswith(".cpp") or file.endswith(".c") or \
         file.endswith(".h") or file.endswith(".hpp"):
        dprint(file)

        # exclude unittest itself
        if (re.match("^unittest\/", root[len(path)+1:])):
          continue
        # exclude files from build directory (auto generated)
        if (re.match("^build\/", root[len(path)+1:])):
          continue
        # exclude CMake artifacts
        if file.startswith("CMakeCCompilerId") or file.startswith("CMakeCXXCompilerId"):
          continue

        # (-1, -1) means untracked file
        targetFiles[os.path.join(root, file)[len(path)+1:]] = (-1, -1)
        dprint("Registered: " + os.path.join(root, file)[len(path)+1:])


  # From the begging, read each line and process "targetFiles"
  parserStatus = 0 # Nothing / Completed Session
  parsingForFile = ""
  lastlines = 0
  for line in out.splitlines():
    m = re.match("File '(.+)'$", line)
    if m:
      if parserStatus == 1:
        sys.exit("[CRITIAL BUG] Status Mismatch: need to be 0")

      parsingForFile = m.group(1)
      if parsingForFile not in targetFiles:
        if re.match("^CMakeCCompilerId", parsingForFile): # ignore cmake artifacts
          continue
        if re.match("^CMakeCXXCompilerId", parsingForFile): # ignore cmake artifacts
          continue
        print("[CRITICAL BUG] Hey! File " + parsingForFile + " is not being found?")
        targetFiles[parsingForFile] = (-1, -1)
      elif targetFiles[parsingForFile] == (-1, -1):
        dprint("Matching new file: " + parsingForFile)
      else:
        dprint("Duplicated file: " + parsingForFile)

      parserStatus = 1 # File name parsed
      continue

    m = re.match("Lines executed:(\d+.\d+)% of (\d+)$", line)
    if m:
      if parserStatus == 0:
        continue
      if parserStatus == 2:
        sys.exit("[CRITICAL BUG] Status Mismatch: need to be 1")
      parserStatus = 2

      rate = float(m.group(1))
      lines = int(m.group(2))

      if parsingForFile not in targetFiles:
        sys.exit("[CRITICAL BUG] targetFiles broken: not found: " + parsingForFile)
      (oldrate, oldlines) = targetFiles[parsingForFile]

      if oldlines == -1: # new instancfe
        targetFiles[parsingForFile] = (rate, lines)
      elif lines == oldlines and rate > oldrate: # overwrite
        targetFiles[parsingForFile] = (rate, lines)
        # anyway, in this mechanis, this can't happen
        sys.exit("[CRITICAL BUG] file " + parsingForFile + " occurs twice??? case 1")
      else:
        sys.exit("[CRITICAL BUG] file " + parsingForFile + " occurs twice??? case 2")
      continue

    if re.match("Creating '", line):
      if parserStatus == 1:
        sys.exit("[CRITICAL BUG] Status mismatch. It should be 0 or 2!")
      parserStatus = 0
      continue

    if re.match("^\s*$", line):
      continue

    sys.exit("[CRITICAL BUG] incorrect gcov output: " + line)

  totalTestedLine = 0
  totalAllLine = 0

  # For each "targetFiles", check if they are covered.
  for filename, (rate, lines) in targetFiles.iteritems():
    if lines == -1: # untracked file
      # CAUTION! wc does line count of untracked files. it counts lines differently
      # TODO: Count lines with the policy of gcov
      linecount = os.popen("wc -l " + os.path.join(path, filename)).read()
      m = re.match("^(\d+)", linecount)
      if not m:
        sys.exit("Cannot read proper wc results for " + filename)
      lines = int(m.group(1))
      rate = 0.0
      print("Untracked File Found!!!")
      print("[" + filename + "] : 0% of " + m.group(1) + " lines")

    totalAllLine += lines
    totalTestedLine += int((lines * rate / 100.0) + 0.5)

  rate = 100.0 * totalTestedLine / totalAllLine
  print("=======================================================")
  print("Lines: " + str(totalAllLine) + "  Covered Rate: " + str(rate) + "%")
  print("=======================================================")

## @brief Do the check for unit test coverage on the given path
#
# @param path The path to be audited
# @return (number of lines counted, ratio of unittested lines)
def check_component(path):
  # Remove last trailing /
  if path[-1:] == '/':
    path = path[:-1]

  buildpath = os.path.join(path, "build")
  searchlimit = 5
  buildpathconst = path

  # If path/build does not exist, try path/../build, path/../../build, ... (limit = 5)
  while ((not os.path.isdir(buildpath)) and searchlimit > 0):
    searchlimit = searchlimit - 1
    buildpathconst = os.path.join(buildpathconst, "..")
    buildpath = os.path.join(buildpathconst, "build")

  # Get gcov report from unittests
  out = os.popen("gcov -p -r -s " + path + " `find " + buildpath +
                 " -name *.gcno`").read()
  dprint(out)
  endpoint = len(out) - 1
  while (out[endpoint] == '\n'):
    endpoint = endpoint - 1
  startpoint = endpoint
  endpoint = endpoint + 1
  while (out[startpoint] != '\n' and startpoint >= 0):
    startpoint = startpoint - 1
  startpoint = startpoint + 1

  lastline = out[startpoint:endpoint]
  m = re.match("Lines executed:(\d+.\d+)% of (\d+)$", lastline)
  if m:
    rate = float(m.group(1))
    lines = int(m.group(2))
  else:
    print("ERROR! Cannot parse gcov result!")
    return (-1, -1)

  return (lines, rate)
  # Call auditEvaders(out, path) if we really become paranoid.

## @brief Check unit test coverage for a specific path. (every code in that path, recursively)
#
# @param The audited path.
def cmd_module(paths):
  lines = 0
  rate = 0
  countrated = 0

  for path in paths:
    (l, rate) = check_component(path)
    lines = lines + l
    countrated = countrated + (rate * l)

  rate = countrated / lines
  if lines < 0:
    return -1

  print("\n\n===========================================================")
  print("Paths for test coverage " + str(paths))
  print(str(lines) + " Lines with " + str(rate) + "% unit test coverage")
  print("===========================================================\n\n\n")
  return 0

countLines = 0
countCoveredLines = 0

## @brief Search for directories containing CMakeLists.txt
#
# @param path The search target
def analyzeEveryFirstCMakeListsTxt(path):
  global countLines, countCoveredLines
  targetName = os.path.join(path, "CMakeLists.txt")
  targetDir = os.path.join(path, "build")

  if os.path.isfile(targetName):
    if os.path.isdir(targetDir):
      (lines, rate) = check_component(path)
      coveredLines = int((rate * float(lines) + 0.5) / 100.0)
      countLines = countLines + lines
      countCoveredLines = countCoveredLines + coveredLines
      print("[ROS Component]" + str(path) + ": " + str(lines) + " Lines with " + str(rate) + "% unit test coverage")
      return 0
    print("[Warning] " + str(path) + " has CMakeLists.txt but not build directory. This may occur if you build with app option")
    return 0

  filenames = os.listdir(path)
  for filename in filenames:
    fullname = os.path.join(path, filename)
    if (os.path.isdir(fullname)):
      analyzeEveryFirstCMakeListsTxt(fullname)
  return 0

## @brief Check all subdirectories with CMakeLists.txt and thier children, skipping subdirectories without it.
#
# @path The search target
def cmd_all(path):
  analyzeEveryFirstCMakeListsTxt(path)
  print("\n\n===========================================================")
  print("Total Lines = " + str(countLines) + " / Covered Lines = " + str(countCoveredLines) + " ( " + str(100.0 * countCoveredLines / countLines) + "% )")
  print("===========================================================\n\n\n")
  return 0

help_messages = {
  'all':
    'python unittestcoverage.py all [PATH to the Audri ROS directory] {additional options}\n'
    '',
  'module':
    'python unittestcoverage.py module [PATH to the component] {additional options}\n'
    '',
  'help':
    'python unittestcoverage.py [command] [command specific options]\n'
    '\n'
    'Comamnds:\n'
    '    all\n'
    '    module\n'
    '    help\n'
    '\n'
    'Additional Options:\n'
    '    -d enable debugprint\n'
    '\n',
}

## @brief Shows the help message
#
# @param command the command line argument
def cmd_help(command=None):
  if (command is None) or (not command):
    command = 'help'
  print(help_messages[command])
  return 0

## @brief The main function
#
def main():
  num = len(sys.argv)
  if num < 2:
    return cmd_help()

  cmd = sys.argv[1]
  args = []

  for arg in sys.argv[2:]:
    if arg == '-d':
      global debugprint
      debugprint = 1
    else:
      args.append(arg)

  if cmd == 'help':
    arg = (sys.argv[2] if num > 2 else None)
    return cmd_help(arg)
  elif cmd == 'all':
    return cmd_all(args)
  elif cmd == 'module':
    return cmd_module(args)

  return cmd_help()

sys.exit(main())
