#!/usr/bin/env bash

failed=0

python generateGoldenTestResult.py
./testcase01.sh $1 || (echo Testcase 01: FAILED; failed=1)


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
