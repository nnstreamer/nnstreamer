#!/bin/sh

set -e

NNS_PKG_NAME=$1
VERSION=$2

tar xzf ${NNS_PKG_NAME}-${VERSION}.tar.gz
rm -f ${NNS_PKG_NAME}-$VERSION.tar.gz
rm -rf ${NNS_PKG_NAME}-$VERSION/api/capi/include/platform
tar czf ${NNS_PKG_NAME}-$VERSION.tar.gz ${NNS_PKG_NAME}-${VERSION}/*
