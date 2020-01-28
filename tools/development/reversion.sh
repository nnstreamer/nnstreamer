#!/bin/bash
paramerr () {
	echo ""
	echo "Error: parameters not given property (need 5 arguments)."
	echo "    \$ $0 old-major old-mid old-minor new-version(full) \"Name <Email>\""
	echo ""
	echo ""
	exit -1
}

if [ "$#" -ne 5 ]; then paramerr; fi

DATE=`date -R`
sed -i "s|^Version:\t$1\.$2\.$3$|Version:\t$4|" packaging/nnstreamer.spec
if [ -n "$5" ]
then
	DATE2=`date -u "+%a %b %d %Y"`
	sed -i "s|^%changelog$|%changelog\n* ${DATE2} $5\n- Release of $4\n|" packaging/nnstreamer.spec
fi

echo $?
sed -i "s|^  version: '$1\.$2\.$3',$|  version: '$4',|" meson.build
echo $?
sed -i "s|^NNSTREAMER_VERSION  := $1\.$2\.$3$|NNSTREAMER_VERSION  := $4|" jni/nnstreamer.mk
echo $?
sed -i "1s|^|nnstreamer ($4.0) unstable xenial bionic; urgency=medium\n\n  * $4 development starts\n\n -- $5  $DATE\n\n|" debian/changelog
echo $?
