#!/usr/bin/env bash

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

set -e

# wget with sha256 check
wget_sha256()
{
  wget -nc "$1"
  if ! (echo "$2 $(basename "$1")" | sha256sum --status -c); then
    echo -e "$1: sha256sum validation failed!"
    exit 1
  fi
}

# create ./model directory if not existed
if [ ! -d model ]; then
    mkdir -p model
fi
# create ./data directory if not existed
if [ ! -d data ]; then
    mkdir -p data
fi

# Download the model if not existed
model_file=./model/Inception-BN.json
params_file=./model/Inception-BN.params
if [ ! -f ${model_file} ] || [ ! -f ${params_file} ]; then
    wget_sha256 http://data.mxnet.io/models/imagenet/inception-bn.tar.gz 9f0d62e4adfde8bf3fef65ba20b3497e7845786b803fa8fed717602b9878ea03
    tar -xvzf inception-bn.tar.gz -C model
    mv ./model/Inception-BN-symbol.json ${model_file}
    mv ./model/Inception-BN-0126.params ${params_file}
fi

# Download ImageNet validation dataset "val_256_q90.rec", which is used by the MXNet community
# The data format is RecordIO https://mxnet.apache.org/versions/1.7/api/faq/recordio
# Also mentioned in: https://mxnet.apache.org/versions/1.6/api/python/docs/tutorials/performance/backend/mkldnn/mkldnn_quantization.html
dataset_file=./data/val_256_q90.rec
if [ ! -f ${dataset_file} ]; then
    cd ./data
    # Warning: it is 1.5GB
    wget_sha256 http://data.mxnet.io/data/val_256_q90.rec 8d90f7c53e2a74dcd08ae072f0247067424545031e6f5deaf57694ccbbe1a79b
    cd ..
fi

# Run test
./simple_test_mxnet || $(echo "Testing Failed!" >&2 && false)
