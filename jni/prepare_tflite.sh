#!/usr/bin/env bash
#currently we are using tensorflow 1.9.0
VERSION="1.9.0"

#Get tensorflow
if [ ! -d "tensorflow-${VERSION}" ]; then
    if [ ! -f "v${VERSION}.tar.gz" ]; then
	echo "[TENSORFLOW-LITE] Download tensorflow-${VERSION}\n"
	wget "https://github.com/tensorflow/tensorflow/archive/v${VERSION}.tar.gz"
	echo "[TENSORFLOW-LITE] Finish Downloading tensorflow-${VERSION}\n"
	echo "[TENSORFLOW-LITE] untar tensorflow-${VERSION}\n"
    fi
    tar xf "v${VERSION}.tar.gz"
fi

if [ ! -d "tensorflow-${VERSION}/tensorflow/contrib/lite/downloads" ]; then
#Download Dependencys
    pushd "tensorflow-${VERSION}"
    echo "[TENSORFLOW-LITE] Download external libraries of tensorflow-${VERSION}\n"
    sed -i "s|flatbuffers/archive/master.zip|flatbuffers/archive/v1.8.0.zip|g" tensorflow/contrib/lite/download_dependencies.sh
    ./tensorflow/contrib/lite/download_dependencies.sh
    popd
fi
