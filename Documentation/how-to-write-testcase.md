# Case 1: gst-launch based test cases (bash unit test)

1. If you don't have a corresponding testcase group, yet (subdirectory in nnstreamer.git/tests/)

  - Create subdirectory in nnstreamer.git/tests/  
  ```  
  $ cd tests
  $ mkdir new_test_group
  ```
  - Create the shell script for the group. It should be named as ```runTest.sh```. 
  - Use the test API scripts located at ```tests/testAPI.sh```.
  - The following is a template:
  ```
  #!/usr/bin/env bash
  source ../testAPI.sh
  
  CASENUMBER1=1
  CASENUMBER2=2
    
  gstTest "your gstreamer pipeline" $CASENUMBER1
  # generating "$CASE1GOLDEN" and "$CASE1RESULT"
  gstTest "your another pipeline" $CASENUMBER2
  # generating "$CASE2GOLDEN" and "$CASE2RESULT"
  
  compareAll $CASE1GOLDEN $CASE1RESULT $CASENUMBER1
  compareAll $CASE2GOLDEN $CASE2RESULT $CASENUMBER2
  
  report
  ```
  - Then the framework will do the rest for you.

2. If you already have a corresponding test case group, add your cases in the corresponding "runTest.sh" in the group subdirectory.


# Case 2: gtest test cases (C/C++ unit test)

If your code is not supposed to be tested via gstreamer CLI pipeline, but with a C/C++ native calls, you are supposed to use gtest framework.

Refer to ```nnstreamer.git/common/test/*``` to write your own gtest test cases.

Then, add your gtest executable to packaging/nnstreamer.spec (refer to the lines with ```unittest_common```) so that we can run the unit test during gbs build.

