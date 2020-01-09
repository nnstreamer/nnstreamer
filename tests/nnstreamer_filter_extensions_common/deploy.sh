#!/bin/bash
##
## @file deploy.sh
## @author Parichay Kapoor <pk.kapoor@gmail.com>
## @date Dec 19 2019
## @brief This creates failure test cases for tensor filter extensions.
BASEPATH=`dirname "$0"`

ext_name=$1
ext_nick_name=$2
model_file=$3
ext_test_gen_dir=$4

pushd $BASEPATH

mkdir -p ${ext_test_gen_dir}
target_file=${ext_test_gen_dir}/unittest_tizen_${ext_name}.cc

if [ ! -f "${target_file}" ]; then
  cp unittest_tizen_template.cc ${target_file}

  sed -i "s|EXT_NAME|${ext_name}|" ${target_file}
  sed -i "s|EXT_NICK_NAME|${ext_nick_name}|" ${target_file}
  sed -i "s|MODEL_FILE|${model_file}|" ${target_file}
fi

popd
