#!/usr/bin/env bash

failed=0
sopath=""

./testcase01.sh $sopath || failed=1

echo ""
echo ""
echo "=================================================="
echo "            Test for tensor_filter_custom"
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
