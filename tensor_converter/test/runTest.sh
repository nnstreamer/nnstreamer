#!/usr/bin/env bash

failed=0
sopath=""

if [ "$1" == "-skipgen" ]
then
  echo "Test Case Generation Skipped"
  sopath=$2
else
  python generateGoldenTestResult.py
  sopath=$1
fi
./testcase01.sh $sopath || failed=1
./testcase02.sh $sopath || failed=1
./testcase03.sh $sopath || failed=1


echo ""
echo ""
echo "=================================================="
echo "            Test for tensor_converter"
echo "=================================================="
if [ "$failed" -eq "0" ]
then
  echo SUCCESS
  echo ""
  exit 0
else
  echo FAILED
  echo ""
  exit -1
fi
