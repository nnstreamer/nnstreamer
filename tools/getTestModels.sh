#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
if [[ $# -eq 0 ]]; then
	TARGET="tflite_model"
else
	TARGET="$1"
fi

mkdir -p ${TARGET}
pushd ${TARGET}
	ln -s ${DIR}/../tests/test_models/models/mobilenet_v1_1.0_224_quant.tflite .
	ln -s ${DIR}/../tests/test_models/labels/labels.txt .
popd
