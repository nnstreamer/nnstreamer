---
title: How to write Test Cases
...

# Case 1: gst-launch based test cases (bash unit test)

1. If you don't have a corresponding testcase group, yet (subdirectory in nnstreamer.git/tests/)

  - Create subdirectory in nnstreamer.git/tests/  
  ```bash
  $ cd tests
  $ mkdir new_test_group
  ```
  - Create the shell script for the group. It should be named as ```runTest.sh```. 
  - Use the test API scripts located at ```ssat-api.sh```.
  - The following is a template:
  ```bash
  #!/usr/bin/env bash
  ##
  ## SPDX-License-Identifier: LGPL-2.1-only
  ##
  ## @file runTest.sh
  ## @author Your Name <your.email@example.com>
  ## @date dd MMM yyyy
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
    
  gstTest "your gst-launch-1.0 Arguments" "test case ID" "set 1 if this is not critical" "set 1 if this passes if gstLaunch fails" "set 1 to enable PERFORMANCE test" "set a positive value (seconds) to enable timeout mode"
  
  report
  ```
  - Then the framework will do the rest for you.

2. If you already have a corresponding test case group, add your cases in the corresponding "runTest.sh" in the group subdirectory.


# Case 2: gtest test cases (C/C++ unit test)

If your code is not supposed to be tested via GStreamer CLI pipeline, but with a C/C++ native calls, you are supposed to use gtest framework.

Refer to ```nnstreamer.git/common/test/*``` to write your own gtest test cases.

Then, add your gtest executable to packaging/nnstreamer.spec (refer to the lines with ```unittest_common```) so that we can run the unit test during gbs build.

