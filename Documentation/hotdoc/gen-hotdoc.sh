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

mkdir -p tmp_doc
cp *.md tmp_doc
cp Documentation/*.md .
cp Documentation/hotdoc/doc-index.md .

sed -i 's+](\.\./+](+g; s+](Documentation/+](+g' *.md
sed -i '\+img src\=\"+s+\./media+Documentation/media+g' *.md
sed -i '\+!\[+s+\./media+Documentation/media+g' *.md

v=$( grep -w version: meson.build | perl -pe 'if(($v)=/([0-9]+([.][0-9]+)+)/){print"$v\n";exit}$_=""' )
deps_file_path="$(pwd)/Documentation/NNStreamer.deps"

echo "NNStreamer version: $v"
echo "Dependencies file path: $deps_file_path"

hotdoc run -i index.md -o Documentation/NNStreamer-doc --sitemap=Documentation/hotdoc/sitemap.txt --deps-file-dest=$deps_file_path \
           --html-extra-theme=Documentation/hotdoc/theme/extra --project-name=NNStreamer --project-version=$v &> hotdoc_result.log

if [[ $? -ne 0 ]]; then
    echo "[ERROR] Failed to run hotdoc. Please check hotdoc_result.log"
fi

rm *.md
cp tmp_doc/*.md .
rm -r tmp_doc
rm -rf Documentation/nnst-exam
