#!/bin/bash
pushd /usr/bin/
wget https://raw.githubusercontent.com/myungjoo/SSAT/main/ssat.sh
wget https://raw.githubusercontent.com/myungjoo/SSAT/main/ssat-api.sh
chmod 755 ssat*.sh
ln -s ssat.sh ssat
popd

mkdir -p /tmp/bmp2png
pushd /tmp/bmp2png
wget https://raw.githubusercontent.com/myungjoo/SSAT/main/util/bmp2png.c
wget https://raw.githubusercontent.com/myungjoo/SSAT/main/util/meson.build
meson build
ninja -C build
cp build/bmp2png /usr/bin
popd
