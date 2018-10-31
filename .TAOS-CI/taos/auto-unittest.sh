#!/usr/bin/env bash

##
# @file auto-unittest.sh
# @brief Auto-generate unit test result
# @how to append this script to /etc/crontab
#       $ sudo vi /etc/crontab
#       30 * * * * www-data /var/www/html/nnstreamer/ci/taos/auto-unittest.sh
# @see      https://github.com/nnsuite/nnstreamer
# @author   Sewon Oh <sewon.oh@samsung.com>
# @param    None
#

# Set-up environements
dirpath="$( cd "$( dirname "$0")" && pwd )"
build_root="${dirpath}/../../"
arch_type="x86_64 armv7l aarch64"
echo -e "[DEBUG] dirpath : $dirpath"
echo -e "[DEBUG] build_root : $build_root"
source ./common/api_collection.sh

# Check dependency
check_dependency gbs
check_dependency mkdir
check_dependency mv
check_dependency rm


# Create result folder
pushd ${build_root}
if [[ -d ci/unittest_result ]]; then
    rm -rf ci/unittest_result/*
else
    mkdir -p ci/unittest_result
fi
for arch in $arch_type
do
    # Gbs build for unit test
    # Unit test for ${arch}
    gbs build -A ${arch} --overwrite --clean > temp.txt

    # Parsing result 
    test_flag=0
    while IFS='' read -r line || [[ -n "$line" ]]; do
        if [[ $line =~  "./unittest_common" ]]; then
            test_flag=1
        fi

        if [[ $line =~ "./testAll.sh" ]]; then
            test_flag=2
            mv result.txt ci/unittest_result/unit_test_common_result_${arch}.txt
        fi

        if [[ $line =~ "popd" ]]; then
            test_flag=0
        fi

        if [[ $test_flag -ne 0 ]]; then
            echo "$line" >> result.txt
        fi
    done < temp.txt
    mv result.txt ci/unittest_result/unit_test_case_result_${arch}.txt   
done
rm temp.txt
popd
