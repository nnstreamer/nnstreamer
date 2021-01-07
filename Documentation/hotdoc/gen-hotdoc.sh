#!/usr/bin/env bash

# This is to create the nnstreamer documents (https://nnstreamer.github.io/).
# Our documentation uses hotdoc, you should usually refer to here: http://hotdoc.github.io/ .
# Run this script on the root path of the NNStreamer.

echo "Generate NNStreamer documents"

if [ -d "nnst-exam" ]
then
    echo "NNStreamer-example repository exists."
else
    echo "Clone NNStreamer-example repository."
    git clone https://github.com/nnstreamer/nnstreamer-example.git Documentation/nnst-exam
fi

v=$( grep -w version: meson.build | perl -pe 'if(($v)=/([0-9]+([.][0-9]+)+)/){print"$v\n";exit}$_=""' )
deps_file_path="$(pwd)/Documentation/NNStreamer.deps"

echo "NNStreamer version: $v"
echo "Dependencies file path: $deps_file_path"

hotdoc run -i index.md -o Documentation/NNStreamer-doc --sitemap=Documentation/hotdoc/sitemap.txt --deps-file-dest=$deps_file_path \
           --html-extra-theme=Documentation/hotdoc/theme/extra --project-name=NNStreamer --project-version=$v

rm -rf Documentation/nnst-exam
