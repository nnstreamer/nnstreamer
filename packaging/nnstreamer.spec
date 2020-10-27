# Execute gbs with --define "testcoverage 1" in case that you must get unittest coverage statistics
%define		gstpostfix	gstreamer-1.0
%define		gstlibdir	%{_libdir}/%{gstpostfix}
%define		nnstexampledir	/usr/lib/nnstreamer/bin
%define		tensorflow_support 0
%define		tensorflow_lite_support	1
%define		tensorflow2_lite_support 1
%define		armnn_support 0
%define		vivante_support 0
%define		flatbuf_support 1
%define		protobuf_support 1
%define		nnfw_support 1
%define		check_test 1
%define		enable_tizen_privilege 1
%define		enable_tizen_feature 1
%define		enable_extra_subplugins 1
%define		package_test 1

%if "%{?profile}" == "tv"
%define		enable_extra_subplugins 0
%define		enable_tizen_privilege 0
%define		check_test 0
%endif

%if 0%{tizen_version_major} >= 5
%define		python_support 1
%else
%define		python_support 0
%endif

%if 0%{tizen_version_major} >= 6
%define		tizen_sensor_support 1
%define		mvncsdk2_support 1
%define		openvino_support 1

%ifarch aarch64 x86_64
# This supports 64bit systems only
%define		edgetpu_support 1
%else
%define		edgetpu_support 0
%endif

%else
%define		tizen_sensor_support 0
%define		mvncsdk2_support 0
%define		edgetpu_support 0
%endif

%define		pytorch_support 0
%define		caffe2_support 0

%ifnarch %arm aarch64
%define 	armnn_support 0
%endif

%ifnarch %arm aarch64
%define		vivante_support 0
%endif

# DA requested to remove unnecessary module builds
%if 0%{?_with_da_profile}
%define	mvncsdk2_support 0
%define	edgetpu_support 0
%define	openvino_support 0
%endif

%if !0%{?enable_extra_subplugins}
%define		protobuf_support 0
%define		python_support 0
%define		mvncsdk2_support 0
%define		edgetpu_support 0
%endif

%if !0%{?check_test}
%define package_test 0
%endif

# If it is tizen, we can export Tizen API packages.
%bcond_with tizen

Name:		nnstreamer
Summary:	gstreamer plugins for neural networks
# Synchronize the version information among Ubuntu, Tizen, Android, and Meson.
# 1. Ubuntu : ./debian/changelog
# 2. Tizen  : ./packaging/nnstreamer.spec
# 3. Android: ./jni/nnstreamer.mk
# 4. Meson  : ./meson.build
Version:	1.7.0
Release:	0
Group:		Applications/Multimedia
Packager:	MyungJoo Ham <myungjoo.ham@samsung.com>
License:	LGPL-2.1
Source0:	nnstreamer-%{version}.tar.gz
Source1:	generate-tarball.sh
Source1001:	nnstreamer.manifest
%if %{with tizen}
Source1002:	capi-nnstreamer.manifest
%endif

## Define build requirements ##
Requires:	gstreamer >= 1.8.0
BuildRequires:	gstreamer-devel
BuildRequires:	gst-plugins-base-devel
BuildRequires:	glib2-devel
BuildRequires:	meson >= 0.50.0

# To run test cases, we need gst plugins
BuildRequires:	gst-plugins-good
%if 0%{tizen_version_major} >= 5
BuildRequires:	gst-plugins-good-extra
%endif
BuildRequires:	gst-plugins-base
# and gtest
BuildRequires:	gtest-devel
# a few test cases uses python
%if 0%{?check_test}
BuildRequires:	python
BuildRequires:	python-numpy
%endif
%if 0%{?python_support}
# for python custom filters
BuildRequires:	pkgconfig(python2)
BuildRequires:	python-numpy-devel
%endif
# Testcase requires bmp2png, which requires libpng
BuildRequires:  pkgconfig(libpng)
%if 0%{?flatbuf_support}
# for flatbuffers
BuildRequires: flatbuffers-devel
%endif
%if 0%{?tensorflow_lite_support}
# for tensorflow-lite
BuildRequires: tensorflow-lite-devel
%endif
%if 0%{?tensorflow2_lite_support}
# for tensorflow2-lite
BuildRequires: tensorflow2-lite-devel
%endif
# custom_example_opencv filter requires opencv-devel
BuildRequires: opencv-devel
# For './testAll.sh' time limit.
BuildRequires: procps
# for protobuf
%if 0%{?protobuf_support}
BuildRequires: protobuf-devel >= 3.4.0
%endif
# for tensorflow
%if 0%{?tensorflow_support}
BuildRequires: tensorflow
BuildRequires: tensorflow-devel
%endif
# for armnn
%if 0%{?armnn_support}
BuildRequires: armnn-devel
BuildRequires:  libarmcl
BuildConflicts: libarmcl-release
%endif

%if 0%{?edgetpu_support}
BuildRequires:	pkgconfig(edgetpu)
%endif

%if 0%{?testcoverage}
# to be compatible with gcc-9, lcov should have a higher version than 1.14.1
BuildRequires: lcov
# BuildRequires:	taos-ci-unittest-coverage-assessment
%endif

%if 0%{mvncsdk2_support}
BuildRequires:	pkgconfig(libmvnc)
%endif

%if 0%{openvino_support}
BuildRequires:	pkgconfig(openvino)
%endif

# for Vivante
# TODO: dann and opencv will be removed in the near future.
%if 0%{?vivante_support}
BuildRequires:  pkgconfig(opencv)
BuildRequires:  pkgconfig(dann)
BuildRequires:  pkgconfig(ovxlib)
BuildRequires:  pkgconfig(amlogic-vsi-npu-sdk)
%endif

%if %{with tizen}
BuildRequires:	pkgconfig(dpm)
%if 0%{tizen_version_major} >= 5
%if 0%{?enable_tizen_privilege}
BuildRequires:	pkgconfig(mm-resource-manager)
%endif
%endif
BuildRequires:	pkgconfig(mm-camcorder)
%if 0%{?enable_tizen_privilege}
BuildRequires:	pkgconfig(capi-privacy-privilege-manager)
%endif
BuildRequires:	pkgconfig(capi-system-info)
BuildRequires:	pkgconfig(capi-base-common)
BuildRequires:	pkgconfig(dlog)
BuildRequires:	gst-plugins-bad-devel
BuildRequires:	gst-plugins-base-devel
# For tizen sensor support
BuildRequires:	pkgconfig(sensor)
BuildRequires:	capi-system-sensor-devel

%ifarch %arm aarch64
%if 0%{?nnfw_support}
# Tizen 5.5 M2+ support nn-runtime (nnfw)
# As of 2019-09-24, unfortunately, nnfw does not support pkg-config
BuildRequires:  nnfw-devel
BuildRequires:  libarmcl
BuildConflicts: libarmcl-release
BuildRequires:  json-glib-devel
%endif
%endif
%endif  # tizen

# Unit Testing Uses SSAT (hhtps://github.com/myungjoo/SSAT.git)
%if 0%{?unit_test}
BuildRequires:	ssat >= 1.1.0
%endif

# For ORC (Oil Runtime Compiler)
BuildRequires:	pkgconfig(orc-0.4)

# Note that debug packages generate an additional build and storage cost.
# If you do not need debug packages, run '$ gbs build ... --define "_skip_debug_rpm 1"'.

%if "%{?_skip_debug_rpm}" == "1"
%global debug_package %{nil}
%global __debug_install_post %{nil}
%endif

## Define Packages ##
%description
NNStreamer is a set of gstreamer plugins to support general neural networks
and their plugins in a gstreamer stream.

# for tensorflow
%if 0%{?tensorflow_support}
%package tensorflow
Summary:	NNStreamer TensorFlow Support
Requires:	nnstreamer = %{version}-%{release}
Requires:	tensorflow
%description tensorflow
NNStreamer's tensor_fliter subplugin of TensorFlow.
It uses C-API of tensorflow, which is not yet stable as of 1.1x.
Thus, the user needs to check the version of Tensorflow with the
Tensorflow used for building this package.
%endif

# for tensorflow-lite
%if 0%{?tensorflow_lite_support}
%package tensorflow-lite
Summary:	NNStreamer TensorFlow Lite Support
Requires:	nnstreamer = %{version}-%{release}
# tensorflow-lite provides .a file and it's embedded into the subplugin. No dep to tflite.
%description tensorflow-lite
NNStreamer's tensor_fliter subplugin of TensorFlow Lite.
%endif

# for tensorflow2-lite
%if 0%{?tensorflow2_lite_support}
%package tensorflow2-lite
Summary:	NNStreamer TensorFlow2 Lite Support
Requires:	nnstreamer = %{version}-%{release}
# tensorflow2-lite provides .a file and it's embedded into the subplugin. No dep to tflite.
%description tensorflow2-lite
NNStreamer's tensor_fliter subplugin of TensorFlow2 Lite.
%endif

%if 0%{?python_support}
%package python2
Summary:  NNStreamer Python Custom Filter Support
Requires: nnstreamer = %{version}-%{release}
Requires: python
%description python2
NNStreamer's tensor_filter subplugin of Python (2.7).
%endif

%if 0%{?armnn_support}
%package armnn
Summary:	NNStreamer Arm NN support
Requires:	armnn
%description armnn
NNStreamer's tensor_filter subplugin of Arm NN Inference Engine.
%endif

# Support vivante subplugin
%if 0%{?vivante_support}
%package vivante
Summary:    NNStreamer subplugin for Verisilion's Vivante
Requires:   nnstreamer = %{version}-%{release}
Requires:   gst-plugins-good
Requires:   gst-plugins-good-extra
Requires:   gst-libav
%description vivante
NNStreamer filter subplugin for Verisicon Vivante.
%define enable_vivante -Denable-vivante=true
%else
%define enable_vivante -Denable-vivante=false
%endif

# for protobuf
%if 0%{?protobuf_support}
%package    protobuf
Summary:	NNStreamer Protobuf Support
Requires:	nnstreamer = %{version}-%{release}
Requires:	protobuf
%description protobuf
NNStreamer's tensor_converter and decoder subplugin of Protobuf.
%endif

# for flatbuf
%if 0%{?flatbuf_support}
%package    flatbuf
Summary:	NNStreamer Flatbuf Support
Requires:	nnstreamer = %{version}-%{release}
Requires:	flatbuffers
%description flatbuf
NNStreamer's tensor_converter and decoder subplugin of flatbuf.
%endif

%package devel
Summary:	Development package for custom tensor operator developers (tensor_filter/custom)
Requires:	nnstreamer = %{version}-%{release}
Requires:	glib2-devel
Requires:	gstreamer-devel
%description devel
Development package for custom tensor operator developers (tensor_filter/custom).
This contains corresponding header files and .pc pkgconfig file.

%package devel-static
Summary:    Static library for nnstreamer-devel package
Requires:   devel = %{version}-%{release}
%description devel-static
Static library package of nnstreamer-devel.

%if 0%{?testcoverage}
%package unittest-coverage
Summary:	NNStreamer UnitTest Coverage Analysis Result
%description unittest-coverage
HTML pages of lcov results of NNStreamer generated during rpmbuild
%endif

%%%% THIS IS FOR TIZEN ONLY! %%%%
%if %{with tizen}
%package -n capi-nnstreamer
Summary:	Tizen Native API for NNStreamer
Group:		Multimedia/Framework
Requires:	%{name} = %{version}-%{release}
%if 0%{tizen_sensor_support}
Requires:	%{name}-tizen-sensor = %{version}-%{release}
%endif
%description -n capi-nnstreamer
Tizen Native API wrapper for NNStreamer.
You can construct a data stream pipeline with neural networks easily.

%post -n capi-nnstreamer -p /sbin/ldconfig
%postun -n capi-nnstreamer -p /sbin/ldconfig

%package nnfw
Summary:	NNStreamer Tizen-nnfw runtime support
Requires:	nnfw
%description nnfw
NNStreamer's tensor_filter subplugin of Tizen-NNFW Runtime. (5.5 M2 +)

%package -n capi-nnstreamer-devel
Summary:	Tizen Native API Devel Kit for NNStreamer
Group:		Multimedia/Framework
Requires:	capi-nnstreamer = %{version}-%{release}
Requires:	capi-ml-common-devel
%description -n capi-nnstreamer-devel
Developmental kit for Tizen Native NNStreamer API.

%package -n capi-nnstreamer-devel-static
Summary:    Static library for Tizen Native API
Group:      Multimedia/Framework
Requires:   capi-nnstreamer-devel = %{version}-%{release}
%description -n capi-nnstreamer-devel-static
Static library of capi-nnstreamer-devel package.

%package -n capi-ml-common-devel
Summary:	Common headers for Tizen Machine Learning API set.
Group:		Multimedia/Framework
%description -n capi-ml-common-devel
Common headers for Tizen Machine Learning API set.

%package -n nnstreamer-tizen-internal-capi-devel
Summary:	Tizen internal API to construct the pipeline
Group:		Multimedia/Framework
Requires:	capi-nnstreamer-devel = %{version}-%{release}
%description -n nnstreamer-tizen-internal-capi-devel
Tizen internal API to construct the pipeline without the permissions.
%endif # tizen

%if 0%{mvncsdk2_support}
%package	ncsdk2
Summary:	NNStreamer Intel Movidius NCSDK2 support
Group:		Multimedia/Framework
%description	ncsdk2
NNStreamer's tensor_fliter subplugin of Intel Movidius Neural Compute stick SDK2.
%endif # mvncsdk2_support

%if 0%{openvino_support}
%package	openvino
Summary:	NNStreamer OpenVino support
Requires:	nnstreamer = %{version}-%{release}
Requires:	openvino
Group:		Multimedia/Framework
%description	openvino
NNStreamer's tensor_fliter subplugin for OpenVino support.
%endif # openvino_support

# Add Tizen's sensor framework API integration
%if 0%{tizen_sensor_support}
%package tizen-sensor
Summary:	NNStreamer integration of tizen sensor framework (tensor_src_tizensensor)
Requires:	nnstreamer = %{version}-%{release}
Requires:	capi-system-sensor
%description tizen-sensor
You can include Tizen sensor framework nodes as source elements of GStreamer/NNStreamer pipelines with this package.
%endif # tizen_sensor_support

%if 0%{?edgetpu_support}
%package edgetpu
Summary:	NNStreamer plugin for Google-Coral Edge TPU
Requires:	libedgetpu1
Requires:	nnstreamer = %{version}-%{release}
%description edgetpu
You may enable this package to use Google Edge TPU with NNStreamer and Tizen ML APIs.
%endif

%if 0%{?package_test}
%package unittests
Summary:	NNStreamer unittests for core, API and plugins
Requires:	nnstreamer = %{version}-%{release}
%description unittests
Various unit test cases and custom subplugin examples for NNStreamer.
%endif

%package util
Summary:	NNStreamer developer utilities
%description util
NNStreamer developer utilities include nnstreamer configuration checker.

## Define build options ##
%define enable_tizen -Denable-tizen=false
%define enable_tizen_sensor -Denable-tizen-sensor=false
%define enable_api -Denable-capi=false
%define enable_mvncsdk2 -Dmvncsdk2-support=disabled
%define enable_openvino -Denable-openvino=false
%define enable_nnfw_runtime -Dnnfw-runtime-support=disabled
%define element_restriction -Denable-element-restriction=false
%define enable_tizen_privilege_check -Denable-tizen-privilege-check=true
%define enable_tizen_feature_check -Denable-tizen-feature-check=true
%define enable_test -Denable-test=true
%define install_test -Dinstall-test=false

%if 0%{mvncsdk2_support}
%define enable_mvncsdk2 -Dmvncsdk2-support=enabled
%endif

%if 0%{openvino_support}
%define enable_openvino -Denable-openvino=true
%endif

%if 0%{tizen_sensor_support}
%define enable_tizen_sensor -Denable-tizen-sensor=true
%endif

%if !0%{?check_test}
%define enable_test -Denable-test=false
%endif

%if 0%{?package_test}
%define install_test -Dinstall-test=true
%endif

%if !0%{?enable_tizen_privilege}
%define enable_tizen_privilege_check -Denable-tizen-privilege-check=false
%endif

%if !0%{?enable_tizen_feature}
%define enable_tizen_feature_check -Denable-tizen-feature-check=false
%endif

%if %{with tizen}
%define enable_tizen -Denable-tizen=true -Dtizen-version-major=0%{tizen_version_major}
%define enable_api -Denable-capi=true
%ifarch %arm aarch64
%if 0%{?nnfw_support}
%define enable_nnfw_runtime -Dnnfw-runtime-support=enabled
%endif  # nnfw_support

%endif
# Element restriction in Tizen
%define restricted_element	'capsfilter input-selector output-selector queue tee valve appsink appsrc audioconvert audiorate audioresample audiomixer videoconvert videocrop videorate videoscale videoflip videomixer compositor fakesrc fakesink filesrc filesink audiotestsrc videotestsrc jpegparse jpegenc jpegdec pngenc pngdec tcpclientsink tcpclientsrc tcpserversink tcpserversrc udpsink udpsrc xvimagesink ximagesink evasimagesink evaspixmapsink glimagesink theoraenc lame vorbisenc wavenc volume oggmux avimux matroskamux v4l2src avsysvideosrc camerasrc tvcamerasrc pulsesrc fimcconvert tizenwlsink gdppay gdpdepay'
%define element_restriction -Denable-element-restriction=true -Drestricted-elements=%{restricted_element}
%endif #if tizen

# Support tensorflow
%if 0%{?tensorflow_support}
%define enable_tf -Dtf-support=enabled
%else
%define enable_tf -Dtf-support=disabled
%endif

# Support tensorflow-lite
%if 0%{?tensorflow_lite_support}
%define enable_tf_lite -Dtflite-support=enabled
%else
%define enable_tf_lite -Dtflite-support=disabled
%endif

# Support tensorflow2-lite
%if 0%{?tensorflow2_lite_support}
%define enable_tf2_lite -Dtflite2-support=enabled
%else
%define enable_tf2_lite -Dtflite2-support=disabled
%endif

# Support pytorch
%if 0%{?pytorch_support}
%define enable_pytorch -Dpytorch-support=enabled
%else
%define enable_pytorch -Dpytorch-support=disabled
%endif

# Support caffe2
%if 0%{?caffe2_support}
%define enable_caffe2 -Dcaffe2-support=enabled
%else
%define enable_caffe2 -Dcaffe2-support=disabled
%endif

# Support ArmNN
%if 0%{?armnn_support}
%define enable_armnn -Darmnn-support=enabled
%else
%define enable_armnn -Darmnn-support=disabled
%endif

# Support python
%if 0%{?python_support}
%define enable_python -Dpython2-support=enabled
%else
%define enable_python -Dpython2-support=disabled
%endif

# Support edgetpu
%if 0%{?edgetpu_support}
%define enable_edgetpu -Denable-edgetpu=true
%else
%define enable_edgetpu -Denable-edgetpu=false
%endif

# Support flatbuffer
%if 0%{?flatbuf_support}
%define enable_flatbuf -Dflatbuf-support=enabled
%else
%define enable_flatbuf -Dflatbuf-support=disabled
%endif

%prep
rm -rf ./build
%setup -q
cp %{SOURCE1001} .
%if %{with tizen}
pushd %{_sourcedir}
sh %{SOURCE1} %{name} %{version}
popd
cp %{SOURCE1002} .
rm -rf ./api/capi/include/platform
%endif

%build
# Remove compiler flags for meson to decide the cpp version
CXXFLAGS=`echo $CXXFLAGS | sed -e "s|-std=gnu++11||"`

%if 0%{?testcoverage}
# To test coverage, disable optimizations (and should unset _FORTIFY_SOURCE to use -O0)
CFLAGS=`echo $CFLAGS | sed -e "s|-O[1-9]|-O0|g"`
CFLAGS=`echo $CFLAGS | sed -e "s|-Wp,-D_FORTIFY_SOURCE=[1-9]||g"`
CXXFLAGS=`echo $CXXFLAGS | sed -e "s|-O[1-9]|-O0|g"`
CXXFLAGS=`echo $CXXFLAGS | sed -e "s|-Wp,-D_FORTIFY_SOURCE=[1-9]||g"`
# also, use the meson's base option, -Db_coverage, instead of --coverage/-fprofile-arcs and -ftest-coverage
%define enable_test_coverage -Db_coverage=true
%else
%define enable_test_coverage -Db_coverage=false
%endif

mkdir -p build

meson --buildtype=plain --prefix=%{_prefix} --sysconfdir=%{_sysconfdir} --libdir=%{_libdir} \
	--bindir=%{nnstexampledir} --includedir=%{_includedir} \
	%{enable_api} %{enable_tizen} %{element_restriction} -Denable-env-var=false -Denable-symbolic-link=false \
	%{enable_tf_lite} %{enable_tf2_lite} %{enable_tf} %{enable_pytorch} %{enable_caffe2} %{enable_python} \
	%{enable_nnfw_runtime} %{enable_mvncsdk2} %{enable_openvino} %{enable_armnn} %{enable_edgetpu}  %{enable_vivante} %{enable_flatbuf} \
	%{enable_tizen_privilege_check} %{enable_tizen_feature_check} %{enable_tizen_sensor} %{enable_test} %{enable_test_coverage} %{install_test} \
	build

ninja -C build %{?_smp_mflags}

export NNSTREAMER_SOURCE_ROOT_PATH=$(pwd)
export NNSTREAMER_BUILD_ROOT_PATH=$(pwd)/build
export GST_PLUGIN_PATH=${NNSTREAMER_BUILD_ROOT_PATH}/gst/nnstreamer:${NNSTREAMER_BUILD_ROOT_PATH}/ext/nnstreamer
export NNSTREAMER_CONF=${NNSTREAMER_BUILD_ROOT_PATH}/nnstreamer-test.ini
export NNSTREAMER_FILTERS=${NNSTREAMER_BUILD_ROOT_PATH}/ext/nnstreamer/tensor_filter
export NNSTREAMER_DECODERS=${NNSTREAMER_BUILD_ROOT_PATH}/ext/nnstreamer/tensor_decoder
export NNSTREAMER_CONVERTERS=${NNSTREAMER_BUILD_ROOT_PATH}/ext/nnstreamer/tensor_converter

%define test_script $(pwd)/packaging/run_unittests_binaries.sh

%if %{with tizen}
    bash %{test_script} ./tests/tizen_nnfw_runtime/unittest_nnfw_runtime_raw
    bash %{test_script} ./tests/tizen_capi/unittest_tizen_sensor
%endif #if tizen
%if 0%{?unit_test}
    bash %{test_script} ./tests
    bash %{test_script} ./tests/tizen_capi/unittest_tizen_capi
    bash %{test_script} ./tests/nnstreamer_filter_extensions_common
    LD_LIBRARY_PATH=${NNSTREAMER_BUILD_ROOT_PATH}/tests/nnstreamer_filter_mvncsdk2:. bash %{test_script} ./tests/nnstreamer_filter_mvncsdk2/unittest_filter_mvncsdk2
%ifarch aarch64 x86_64
    LD_LIBRARY_PATH=${NNSTREAMER_BUILD_ROOT_PATH}/tests/nnstreamer_filter_edgetpu:. bash %{test_script} ./tests/nnstreamer_filter_edgetpu/unittest_edgetpu
%endif #ifarch 64
    pushd tests
    ssat -n --summary summary.txt -cn _n
    popd
%endif #if unit_test

python tools/development/count_test_cases.py build tests/summary.txt

pushd tools/development/confchk
    rm -rf build
    meson --prefix=%{_prefix} build
    ninja -C build
popd

%install
DESTDIR=%{buildroot} ninja -C build %{?_smp_mflags} install
pushd tools/development/confchk
DESTDIR=%{buildroot} ninja -C build install
popd

pushd %{buildroot}%{_libdir}
ln -sf %{gstlibdir}/libnnstreamer.so libnnstreamer.so
popd

%if 0%{?python_support}
mkdir -p %{buildroot}%{python_sitelib}
pushd %{buildroot}%{python_sitelib}
ln -sf %{_prefix}/lib/nnstreamer/filters/nnstreamer_python2.so nnstreamer_python.so
popd
%endif

# Hotfix: Support the backward compatibility of the .Net APIs
pushd %{buildroot}%{_libdir}
ln -sf libcapi-nnstreamer.so.%{version} libcapi-nnstreamer.so.0
popd

%if 0%{?testcoverage}
##
# The included directories are:
#
# api: nnstreamer api
# gst: the nnstreamer elements
# nnstreamer_example: custom plugin examples
# common: common libraries for gst (elements)
# include: common library headers and headers for external code (packaged as "devel")
#
# Intentionally excluded directories are:
#
# tests: We are not going to show testcoverage of the test code itself or example applications

%if %{with tizen}
%define testtarget $(pwd)/api/capi
%else
%define testtarget
%endif

# 'lcov' generates the date format with UTC time zone by default. Let's replace UTC with KST.
# If you ccan get a root privilege, run ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime
TZ='Asia/Seoul'; export TZ

# Get commit info
VCS=`cat ${RPM_SOURCE_DIR}/nnstreamer.spec | grep "^VCS:" | sed "s|VCS:\\W*\\(.*\\)|\\1|"`

# Create human readable unit test coverage report web page.
# Create null gcda files if gcov didn't create it because there is completely no unit test for them.
find . -name "*.gcno" -exec sh -c 'touch -a "${1%.gcno}.gcda"' _ {} \;
# Remove gcda for meaningless file (CMake's autogenerated)
find . -name "CMakeCCompilerId*.gcda" -delete
find . -name "CMakeCXXCompilerId*.gcda" -delete
#find . -path "/build/*.j
# Generate report
lcov -t 'NNStreamer Unit Test Coverage' -o unittest.info -c -d . -b $(pwd) --no-external
# Exclude generated files (Orc) and device-dependent filters.
lcov -r unittest.info "*/*-orc.*" "*/tests/*" "*/meson*/*" "*/*@sha/*" "*/*_openvino*" "*/*_edgetpu*" "*/*_movidius_ncsdk2*" -o unittest-filtered.info
# Visualize the report
genhtml -o result unittest-filtered.info -t "nnstreamer %{version}-%{release} ${VCS}" --ignore-errors source -p ${RPM_BUILD_DIR}

mkdir -p %{buildroot}%{_datadir}/nnstreamer/unittest/
cp -r result %{buildroot}%{_datadir}/nnstreamer/unittest/
%endif  # test coverage

%post -p /sbin/ldconfig

%postun -p /sbin/ldconfig

%files
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
%license LICENSE
%{_prefix}/lib/nnstreamer/decoders/libnnstreamer_decoder_bounding_boxes.so
%{_prefix}/lib/nnstreamer/decoders/libnnstreamer_decoder_pose_estimation.so
%{_prefix}/lib/nnstreamer/decoders/libnnstreamer_decoder_image_segment.so
%{_prefix}/lib/nnstreamer/decoders/libnnstreamer_decoder_image_labeling.so
%{_prefix}/lib/nnstreamer/decoders/libnnstreamer_decoder_direct_video.so
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_cpp.so
%{gstlibdir}/libnnstreamer.so
%{_libdir}/libnnstreamer.so
%{_sysconfdir}/nnstreamer.ini

# for tensorflow
%if 0%{?tensorflow_support}
%files tensorflow
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_tensorflow.so
%endif

%if 0%{?tensorflow_lite_support}
%files tensorflow-lite
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_tensorflow-lite.so
%endif

# for tensorflow2-lite
%if 0%{?tensorflow2_lite_support}
%files tensorflow2-lite
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_tensorflow2-lite.so
%endif

%if 0%{?python_support}
%files python2
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_python2.so
%{_prefix}/lib/nnstreamer/filters/nnstreamer_python2.so
%{python_sitelib}/nnstreamer_python.so
%endif

%if 0%{?protobuf_support}
%files protobuf
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
%{_libdir}/libnnstreamer_protobuf.so
%{_prefix}/lib/nnstreamer/decoders/libnnstreamer_decoder_protobuf.so
%{_prefix}/lib/nnstreamer/converters/libnnstreamer_converter_protobuf.so
%endif

%if 0%{?flatbuf_support}
%files flatbuf
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
%{_prefix}/lib/nnstreamer/decoders/libnnstreamer_decoder_flatbuf.so
%{_prefix}/lib/nnstreamer/converters/libnnstreamer_converter_flatbuf.so
%endif

%files devel
%{_includedir}/nnstreamer/tensor_typedef.h
%{_includedir}/nnstreamer/tensor_filter_custom.h
%{_includedir}/nnstreamer/tensor_filter_custom_easy.h
%{_includedir}/nnstreamer/nnstreamer_plugin_api_filter.h
%{_includedir}/nnstreamer/nnstreamer_plugin_api_decoder.h
%{_includedir}/nnstreamer/nnstreamer_plugin_api_converter.h
%{_includedir}/nnstreamer/nnstreamer_plugin_api.h
%{_includedir}/nnstreamer/nnstreamer_version.h
%{_includedir}/nnstreamer/tensor_filter_cpp.hh
%{_includedir}/nnstreamer/nnstreamer_cppplugin_api_filter.hh
%{_libdir}/pkgconfig/nnstreamer.pc
%{_libdir}/pkgconfig/nnstreamer-cpp.pc

%files devel-static
%{_libdir}/*.a
%exclude %{_libdir}/libcapi*.a

%if 0%{?testcoverage}
%files unittest-coverage
%{_datadir}/nnstreamer/unittest/*
%endif

%if %{with tizen}
%files -n capi-nnstreamer
%manifest capi-nnstreamer.manifest
%license LICENSE
%{_libdir}/libcapi-nnstreamer.so.*

%files -n capi-nnstreamer-devel
%{_includedir}/nnstreamer/nnstreamer.h
%{_includedir}/nnstreamer/nnstreamer-single.h
%{_libdir}/libcapi-nnstreamer.so
%{_libdir}/pkgconfig/capi-nnstreamer.pc

%files -n capi-nnstreamer-devel-static
%{_libdir}/libcapi-nnstreamer.a

%files -n capi-ml-common-devel
%{_includedir}/nnstreamer/ml-api-common.h

%files -n nnstreamer-tizen-internal-capi-devel
%{_includedir}/nnstreamer/nnstreamer-tizen-internal.h

%ifarch %arm aarch64
%if 0%{?nnfw_support}
%files nnfw
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_nnfw.so
%endif
%endif

%if 0%{?armnn_support}
%files armnn
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_armnn.so
%endif
%endif  # tizen

%if 0%{mvncsdk2_support}
%files -n nnstreamer-ncsdk2
%defattr(-,root,root,-)
%manifest nnstreamer.manifest
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_movidius-ncsdk2.so
%endif  # mvncsdk2_support

%if 0%{?vivante_support}
%files vivante
%defattr(-,root,root,-)
%manifest nnstreamer.manifest
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_vivante.so
%endif

%if 0%{tizen_sensor_support}
%files tizen-sensor
%manifest nnstreamer.manifest
%{gstlibdir}/libnnstreamer-tizen-sensor.so
%endif  # tizen_sensor_support

%if 0%{?edgetpu_support}
%files edgetpu
%manifest nnstreamer.manifest
%license LICENSE
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_edgetpu.so
%endif

%if 0%{?package_test}
%files unittests
%manifest nnstreamer.manifest
%{_libdir}/libnnstreamer_unittest_util.so
%{_prefix}/lib/nnstreamer/customfilters/*.so
%{_prefix}/lib/nnstreamer/unittest
%endif

%if 0%{?openvino_support}
%files openvino
%manifest nnstreamer.manifest
%license LICENSE
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_openvino.so
%endif

%files util
%{_bindir}/nnstreamer-check

%changelog
* Wed Sep 09 2020 MyungJoo Ham <myungjoo.ham@samsung.com>
- Start development of 1.7.0 (1.8.0-RC1)

* Wed Sep 09 2020 MyungJoo Ham <myungjoo.ham@samsung.com>
- Release of 1.6.0 (LTS for Tizen 6.0 M2 and Android-next products)

* Thu Jun 04 2020 MyungJoo Ham <myungjoo.ham@samsung.com>
- Start development of 1.5.3 (1.6.0-RC4)

* Wed Apr 08 2020 MyungJoo Ham <myungjoo.ham@samsung.com>
- Start development of 1.5.2 (1.6.0-RC3)

* Wed Mar 18 2020 MyungJoo Ham <myungjoo.ham@samsung.com>
- Start development of 1.5.1 (1.6.0-RC2)

* Tue Feb 11 2020 MyungJoo Ham <myungjoo.ham@samsung.com>
- Start development of 1.5.0 (1.6.0-RC1)

* Tue Feb 11 2020 MyungJoo Ham <myungjoo.ham@samsung.com>
- Release of 1.4.0 (Tensor-filter API has been updated!)

* Mon Feb 03 2020 MyungJoo Ham <myungjoo.ham@samsung.com>
- Release of 1.3.1

* Wed Dec 11 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
- 1.3.0 development starts

* Wed Dec 11 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
- Release of 1.2.0

* Thu Sep 26 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
- Release of 1.0.0 (1.0 RC2 == 1.0 Release for Tizen 5.5 M2)

* Wed Aug 14 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
- Release of 0.3.0 (1.0 RC1)

* Mon May 27 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
- Release of 0.2.0

* Wed Mar 20 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
- Release of 0.1.2

* Mon Feb 25 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
- Release of 0.1.1

* Thu Jan 24 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
- Release of 0.1.0

* Mon Dec 03 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
- Release of 0.0.3

* Mon Oct 15 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
- Started single-binary packaging for 0.0.3

* Fri May 25 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
- Packaged tensor_convert plugin.
