###########################################################################
#
#              Options for gbs/rpmbuild users
#
# gbs -c .TAOS-CI/.gbs.conf build --define "unit_test 1"
#       Execute all unit test cases
#
# gbs -c .TAOS-CI/.gbs.conf build --define "testcoverage 1"
# 	Generate unittest coverage statistics
#       Use with "unit_test 1" to do it with as many cases as possible:
#       $ gbs -c .TAOS-CI/.gbs.conf build --define "unit_test 1" --define "testcoverage 1"
#

%define		gstpostfix	gstreamer-1.0
%define		gstlibdir	%{_libdir}/%{gstpostfix}
%define		nnstbindir	lib/nnstreamer/bin

###########################################################################
# Default features for Tizen releases
# If you want to build RPM for other Linux distro, you may need to
# touch these values for your needs.
%define		tensorflow_support 0
%define		tensorflow_lite_support	1
%define		tensorflow2_lite_support 1
%define		armnn_support 1
%define		vivante_support 0
%define		flatbuf_support 1
%define		protobuf_support 1
%define		nnfw_support 1
%define		grpc_support 1
%define		pytorch_support 0
%define		caffe2_support 0
%define		mqtt_support 1
%define		lua_support 1
%define		tvm_support 1
%define		snpe_support 1
%define		trix_engine_support 1
# Support AI offloading (tensor_query) using nnstreamer-edge interface
%define		nnstreamer_edge_support 1

%define		check_test 1
%define		release_test 1

###########################################################################
# Conditional features for Tizen releases

# Enable python if it's Tizen 5.0+
%if 0%{tizen_version_major} >= 5
%define		python3_support 1
%else
%define		python3_support 0
%endif

# Enable Tizen-Sensor, OpenVINO, MVNCSDK2 if it's Tizen 6.0+
%if 0%{tizen_version_major} >= 6
%define		tizen_sensor_support 1
%define		mvncsdk2_support 1
%define		openvino_support 1
%define		edgetpu_support 1
%else
%define		tizen_sensor_support 0
%define		mvncsdk2_support 0
%define		openvino_support 0
%define		edgetpu_support 0
%endif

# tizen 6.0 (or less) backward-compatibility check
%if ( 0%{?tizen_version_major} == 6 && 0%{?tizen_version_minor} < 5 ) || 0%{?tizen_version_major} < 6
%define		grpc_support 0
%define		tensorflow2_lite_support 0
%define		trix_engine_support 0
%endif

# Disable e-TPU if it's not 64bit system
%ifnarch aarch64 x86_64
%define		edgetpu_support 0
%endif

# Disable ARMNN/Vivante/NNFW if it's not ARM.
%ifnarch %arm aarch64
%define 	armnn_support 0
%define		vivante_support 0
%endif

# Disable NNFW if it's not ARM/x64 and x86
%ifnarch %arm aarch64 x86_64 i586 i686 %ix86
%define		nnfw_support 0
%endif

# Disable a few features for TV releases
%if "%{?profile}" == "tv"
%define		grpc_support 0
%define		check_test 0
%define		edgetpu_support 0
%define		protobuf_support 0
%define		python3_support 0
%define		mvncsdk2_support 0
%define		openvino_support 0
%define		nnfw_support 0
%define		armnn_support 0
%define		vivante_support 0
%define		pytorch_support 0
%define		caffe2_support 0
%define		tensorflow_support 0
%define		lua_support 0
%define		mqtt_support 0
%define		tvm_support 0
%define		snpe_support 0
%define		trix_engine_support 0
%define		nnstreamer_edge_support 0
%endif

# DA requested to remove unnecessary module builds
%if 0%{?_with_da_profile}
%define		mvncsdk2_support 0
%define		edgetpu_support 0
%define		openvino_support 0
%define		edgetpu_support 0
%define		armnn_support 0
%define		lua_support 0
%define		mqtt_support 0
%define		tvm_support 0
%define		trix_engine_support 0
%endif

# Release unit test suite as a subpackage only if check_test is enabled.
%if !0%{?check_test}
%define		release_test 0
%endif

# Current Tizen Robot profile only supports aarch64.
%ifnarch aarch64
%define		snpe_support 0
%endif

# If it is tizen, we can export Tizen API packages.
%bcond_with tizen

###########################################################################
# Package / sub-package definitions
Name:		nnstreamer
Summary:	gstreamer plugins for neural networks
# Synchronize the version information among Ubuntu, Tizen, Android, and Meson.
# 1. Ubuntu : ./debian/changelog
# 2. Tizen  : ./packaging/nnstreamer.spec
# 3. Android: ./jni/nnstreamer.mk
# 4. Meson  : ./meson.build
Version:	2.3.0
Release:	0
Group:		Machine Learning/ML Framework
Packager:	MyungJoo Ham <myungjoo.ham@samsung.com>
License:	LGPL-2.1
Source0:	nnstreamer-%{version}.tar
Source1001:	nnstreamer.manifest

## Define requirements ##
Requires: nnstreamer-core = %{version}-%{release}
Requires: nnstreamer-configuration = %{version}-%{release}
Requires: nnstreamer-single = %{version}-%{release}
Recommends: nnstreamer-default-configuration = %{version}-%{release}
%if 0%{?nnstreamer_edge_support}
BuildRequires: nnstreamer-edge-devel
%endif

## Define build requirements ##
BuildRequires:	gstreamer-devel
BuildRequires:	gst-plugins-base-devel
BuildRequires:	gst-plugins-bad-devel
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
BuildRequires:	python3
%endif
%if 0%{?python3_support}
# for python3 custom filters
BuildRequires:	python3-devel
BuildRequires:	python3-numpy-devel
%endif
# Testcase requires bmp2png, which requires libpng
BuildRequires:  pkgconfig(libpng)
%if 0%{?flatbuf_support}
# for flatbuffers
BuildRequires: flatbuffers-devel
%if 0%{?unit_test}
BuildRequires: flatbuffers-python
%endif
%endif
%if 0%{?tensorflow_lite_support}
# for tensorflow-lite
BuildRequires: tensorflow-lite-devel
%endif
%if 0%{?tensorflow2_lite_support}
# for tensorflow2-lite
BuildRequires: tensorflow2-lite-devel
# tensorflow2-lite-custom requires scripts for rpm >= 4.9
BuildRequires:  rpm >= 4.9
%global __requires_exclude ^libtensorflow2-lite-custom.*$
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

%if 0%{?grpc_support}
BuildRequires:  grpc-devel
%endif

%if %{with tizen}
BuildRequires:	pkgconfig(dlog)
# For tizen sensor support
BuildRequires:	pkgconfig(sensor)
BuildRequires:	capi-system-sensor-devel
%endif  # tizen

%if 0%{?nnfw_support}
# Tizen 5.5 M2+ support nn-runtime (nnfw)
# As of 2019-09-24, unfortunately, nnfw does not support pkg-config
BuildRequires:  nnfw-devel
%ifarch %arm aarch64
BuildRequires:  libarmcl
BuildConflicts: libarmcl-release
%endif
%endif

%if 0%{?pytorch_support}
BuildRequires:	pytorch-devel
%endif

# Caffe2 is merged to pytorch
%if 0%{?caffe2_support}
BuildRequires:	pytorch-devel
%endif

%if 0%{?lua_support}
BuildRequires:	lua-devel
%endif

%if 0%{?tvm_support}
BuildRequires:	tvm-runtime-devel
%endif

%if 0%{?snpe_support}
BuildRequires:	snpe-devel
%endif

%if 0%{?trix_engine_support}
BuildRequires:	npu-engine-devel
%endif

# Unit Testing Uses SSAT (https://github.com/myungjoo/SSAT.git)
%if 0%{?unit_test} || 0%{?edge_test}
BuildRequires:	ssat >= 1.1.0
# Mosquitto MQTT broker for unit testing
BuildRequires:  mosquitto
%endif

# For ORC (Oil Runtime Compiler)
BuildRequires:	pkgconfig(orc-0.4)

# For nnstreamer-parser
BuildRequires:	flex
BuildRequires:	bison

# Note that debug packages generate an additional build and storage cost.
# If you do not need debug packages, run '$ gbs -c .TAOS-CI/.gbs.conf build ... --define "_skip_debug_rpm 1"'.

%if "%{?_skip_debug_rpm}" == "1"
%global debug_package %{nil}
%global __debug_install_post %{nil}
%endif

## Define Packages ##
%description
NNStreamer is a set of gstreamer plugins to support general neural networks
and their plugins in a gstreamer stream. NNStreamer is a meta package of
nnstreamer-core and nnstreamer-configuration

%package core
Requires: gstreamer >= 1.8.0
%if 0%{?nnstreamer_edge_support}
Requires: nnstreamer-edge
%endif
Summary: NNStreamer core package
%description core
NNStreamer is a set of gstreamer plugins to support general neural networks
and their plugins in a gstreamer stream, this package is core package without configuration

%package single
Summary: NNStreamer singe-shot package
%description single
Element to use general neural network framework directly without gstreamer pipeline.

%package default-configuration
Summary: NNStreamer global configuration
Provides: nnstreamer-configuration = %{version}-%{release}
Conflicts: nnstreamer-test-devel
%description default-configuration
NNStreamer's global configuration setup for the end user.

# for tensorflow
%if 0%{?tensorflow_support}
%package tensorflow
Summary:	NNStreamer TensorFlow Support
Requires:	nnstreamer = %{version}-%{release}
Requires:	tensorflow
%description tensorflow
NNStreamer's tensor_filter subplugin of TensorFlow.
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
NNStreamer's tensor_filter subplugin of TensorFlow Lite.
%endif

# for tensorflow2-lite
%if 0%{?tensorflow2_lite_support}
%package tensorflow2-lite
Summary:	NNStreamer TensorFlow2 Lite Support
Requires:	nnstreamer = %{version}-%{release}
# tensorflow2-lite provides .a file and it's embedded into the subplugin. No dep to tflite.
%description tensorflow2-lite
NNStreamer's tensor_filter subplugin of TensorFlow2 Lite.
%endif

%if 0%{?python3_support}
%package python3
Summary:  NNStreamer Python3 Custom Filter Support
Requires: nnstreamer = %{version}-%{release}
%description python3
NNStreamer's tensor_filter subplugin of Python3.
%endif

%if 0%{?armnn_support}
%package armnn
Summary:	NNStreamer Arm NN support
Requires:	nnstreamer = %{version}-%{release}
Requires:	armnn
%description armnn
NNStreamer's tensor_filter subplugin of Arm NN Inference Engine.
%endif

# Support vivante subplugin
%if 0%{?vivante_support}
%package vivante
Summary:    NNStreamer subplugin for Verisilicon's Vivante
Requires:   nnstreamer = %{version}-%{release}
%description vivante
NNStreamer filter subplugin for Verisilicon Vivante.
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
%if "%{?profile}" != "tv"
Recommends: flatbuffers-python
%endif
%description flatbuf
NNStreamer's tensor_converter and decoder subplugin of flatbuf.
%endif

# for pytorch
%if 0%{?pytorch_support}
%package pytorch
Summary:	NNStreamer PyTorch Support
Requires:	nnstreamer = %{version}-%{release}
Requires:	pytorch
%description pytorch
NNStreamer's tensor_filter subplugin of pytorch
%endif

# for caffe2
%if 0%{?caffe2_support}
%package caffe2
Summary:	NNStreamer caffe2 Support
Requires:	nnstreamer = %{version}-%{release}
Requires:	pytorch
%description caffe2
NNStreamer's tensor_filter subplugin of caffe2
%endif

# for lua
%if 0%{?lua_support}
%package lua
Summary:	NNStreamer lua Support
Requires:	nnstreamer = %{version}-%{release}
Requires:	lua
%description lua
NNStreamer's tensor_filter subplugin of lua
%endif

%if 0%{?tvm_support}
%package tvm
Summary:	NNStreamer TVM support
Requires:	nnstreamer = %{version}-%{release}
Requires:	tvm
%description tvm
NNStreamer's tensor_filter subplugin of tvm
%endif

# for snpe
%if 0%{?snpe_support}
%package snpe
Summary:	NNStreamer snpe Support
Requires:	nnstreamer = %{version}-%{release}
Requires:	snpe
%description snpe
NNStreamer's tensor_filter subplugin of snpe
%endif

# for trix-engone
%if 0%{?trix_engine_support}
%package trix-engine
Summary:	NNStreamer TRIx-Engine support
Requires:	nnstreamer = %{version}-%{release}
Requires:	trix-engine
%description trix-engine
NNStreamer's tensor_filter subplugin of trix-engine
%endif

%package devel
Summary:	Development package for custom tensor operator developers (tensor_filter/custom)
Requires:	nnstreamer = %{version}-%{release}
Requires:	nnstreamer-single-devel = %{version}-%{release}
Requires:	glib2-devel
Requires:	gstreamer-devel
%description devel
Development package for subplugin or custom filter developers.
Developers may add support for new hardware accelerators or neural network
frameworks, or introduce new data types and their converters for tensors.
However, applications or service developers generally do not need this.
This contains corresponding header files and .pc pkgconfig file.

%package devel-internal
Summary:    Development package to access internal functions of NNStreamer
Requires:   nnstreamer-devel = %{version}-%{release}
%description devel-internal
Development package to access internal functions of NNStreamer.
This may be used by API packages, which wrap nnstreamer features.
In most cases, custom-filter or subplugin authors do not need this internal devel package; however, if they want to access more internal functions, they may need this.

%package devel-static
Summary:    Static library for nnstreamer-devel package
Requires:   nnstreamer-devel = %{version}-%{release}
%description devel-static
Static library package of nnstreamer-devel.

%package single-devel
Summary:    Development package for NNStreamer single-shot
Requires:   nnstreamer-single = %{version}-%{release}
%description single-devel
Development package for NNStreamer single-shot.

%package single-devel-static
Summary:    Static library for nnstreamer-single-devel package
Requires:   nnstreamer-single-devel = %{version}-%{release}
%description single-devel-static
Static library package of nnstreamer-single-devel.

%package test-devel
Summary: Development package to provide testable environment of a subplugin (tensor_filter/custom)
Requires: nnstreamer-devel = %{version}-%{release}
Requires: nnstreamer-single-devel = %{version}-%{release}
Provides: nnstreamer-configuration = %{version}-%{release}
Conflicts: nnstreamer-default-configuration
%description test-devel
Development package to provide testable environment of NNStreamer sub-plugin.
This package enables testable environment of NNStreamer sub-plugin by making nnstreamer to recognize NNSTREAMER_CONF_PATH to steer a sub-plugin path to a custom path.
Also This package provides test templates to be used with.

%if 0%{?testcoverage}
%package unittest-coverage
Summary:	NNStreamer UnitTest Coverage Analysis Result
%description unittest-coverage
HTML pages of lcov results of NNStreamer generated during rpmbuild
%endif

%if 0%{?nnfw_support}
%package nnfw
Summary:	NNStreamer Tizen-nnfw runtime support
Requires:	nnstreamer = %{version}-%{release}
Requires:	nnfw
%description nnfw
NNStreamer's tensor_filter subplugin of Tizen-NNFW Runtime. (5.5 M2 +)
%endif

%if 0%{?mvncsdk2_support}
%package	ncsdk2
Summary:	NNStreamer Intel Movidius NCSDK2 support
Requires:	nnstreamer = %{version}-%{release}
Group:		Machine Learning/ML Framework
%description	ncsdk2
NNStreamer's tensor_filter subplugin of Intel Movidius Neural Compute stick SDK2.
%endif # mvncsdk2_support

%if 0%{openvino_support}
%package	openvino
Summary:	NNStreamer OpenVino support
Requires:	nnstreamer = %{version}-%{release}
Requires:	openvino
Group:		Machine Learning/ML Framework
%description	openvino
NNStreamer's tensor_filter subplugin for OpenVino support.
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

%if 0%{grpc_support}
%package grpc
Summary:	NNStreamer gRPC support
Requires:	nnstreamer = %{version}-%{release}
%if %{with tizen}
Recommends:	nnstreamer-grpc-protobuf = %{version}-%{release}
Recommends:	nnstreamer-grpc-flatbuf = %{version}-%{release}
%endif
%description  grpc
NNStreamer's tensor_source/sink plugins for gRPC support.

%if 0%{protobuf_support}
%package grpc-protobuf
Summary:	NNStreamer gRPC/Protobuf support
Requires:	nnstreamer-grpc = %{version}-%{release}
Requires:	nnstreamer-protobuf = %{version}-%{release}
%description  grpc-protobuf
NNStreamer's gRPC IDL support for protobuf
%endif

%if 0%{flatbuf_support}
%package grpc-flatbuf
Summary:	NNStreamer gRPC/Flatbuf support
Requires:	nnstreamer-grpc = %{version}-%{release}
Requires:	nnstreamer-flatbuf = %{version}-%{release}
%description  grpc-flatbuf
NNStreamer's gRPC IDL support for flatbuf
%endif

%endif # grpc_support

%if 0%{?edgetpu_support}
%package edgetpu
Summary:	NNStreamer plugin for Google-Coral Edge TPU
Requires:	libedgetpu1
Requires:	nnstreamer = %{version}-%{release}
%description edgetpu
You may enable this package to use Google Edge TPU with NNStreamer and Tizen ML APIs.
%endif

%if 0%{?release_test}
%package unittests
Summary:	NNStreamer unittests for core and plugins
Requires:	nnstreamer = %{version}-%{release}
%description unittests
Various unit test cases and custom subplugin examples for NNStreamer.
%endif

%package util
Summary:	NNStreamer developer utilities
%description util
NNStreamer developer utilities include nnstreamer configuration checker.

%package misc
Summary:	NNStreamer extra packages
%if 0%{?mqtt_support}
BuildRequires:	pkgconfig(paho-mqtt-c)
%endif

%description misc
Provides additional gstreamer plugins for nnstreamer pipelines

## Define build options ##
%define enable_tizen -Denable-tizen=false
%define enable_tizen_sensor -Denable-tizen-sensor=false
%define enable_mvncsdk2 -Dmvncsdk2-support=disabled
%define enable_openvino -Denable-openvino=false
%define enable_nnfw_runtime -Dnnfw-runtime-support=disabled
%define element_restriction -Denable-element-restriction=false
%define enable_test -Denable-test=true
%define install_test -Dinstall-test=false

%if 0%{mvncsdk2_support}
%define enable_mvncsdk2 -Dmvncsdk2-support=enabled
%endif

%if 0%{openvino_support}
%define enable_openvino -Denable-openvino=true
%endif

%if 0%{?nnfw_support}
%define enable_nnfw_runtime -Dnnfw-runtime-support=enabled
%endif  # nnfw_support

%if 0%{tizen_sensor_support}
%define enable_tizen_sensor -Denable-tizen-sensor=true
%endif

%if !0%{?check_test}
%define enable_test -Denable-test=false
%endif

%if 0%{?release_test}
%define install_test -Dinstall-test=true
%endif

%if %{with tizen}
%define enable_tizen -Denable-tizen=true -Dtizen-version-major=0%{tizen_version_major}
# Element allowance in Tizen
%define allowed_element_base     'capsfilter input-selector output-selector queue tee valve appsink appsrc audioconvert audiorate audioresample audiomixer videoconvert videocrop videorate videoscale videoflip videomixer compositor fakesrc fakesink filesrc filesink audiotestsrc videotestsrc jpegparse jpegenc jpegdec pngenc pngdec tcpclientsink tcpclientsrc tcpserversink tcpserversrc xvimagesink ximagesink evasimagesink evaspixmapsink glimagesink theoraenc lame vorbisenc wavenc volume oggmux avimux matroskamux v4l2src avsysvideosrc camerasrc tvcamerasrc pulsesrc fimcconvert tizenwlsink gdppay gdpdepay join '
%define allowed_element_edgeai   'rtpdec rtspsrc rtspclientsink zmqsrc zmqsink mqttsrc mqttsink udpsrc udpsink multiudpsink edgesrc edgesink '
%define allowed_element_audio    'audioamplify audiochebband audiocheblimit audiodynamic audioecho audiofirfilter audioiirfilter audioinvert audiokaraoke audiopanorama audiowsincband audiowsinclimit scaletempo stereo '
%if "%{?profile}" == "tv"
%define allowed_element_vd       'tvdpbsrc '
%define allowed_element          %{allowed_element_base}%{allowed_element_audio}%{allowed_element_edgeai}%{allowed_element_vd}
%else
%define allowed_element          %{allowed_element_base}%{allowed_element_audio}%{allowed_element_edgeai}
%endif
%define element_restriction -Denable-element-restriction=true -Dallowed-elements=%{allowed_element}
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
%define enable_tf2_lite -Dtflite2-support=enabled -Dtflite2-custom-support=enabled
%else
%define enable_tf2_lite -Dtflite2-support=disabled -Dtflite2-custom-support=disabled
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
%if 0%{?python3_support}
%define enable_python3 -Dpython3-support=enabled
%else
%define enable_python3 -Dpython3-support=disabled
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

# Support mqtt
%if 0%{?mqtt_support}
%define enable_mqtt -Dmqtt-support=enabled
%else
%define enable_mqtt -Dmqtt-support=disabled
%endif

# Support lua
%if 0%{?lua_support}
%define enable_lua -Dlua-support=enabled
%else
%define enable_lua -Dlua-support=disabled
%endif

# Support tvm
%if 0%{?tvm_support}
%define enable_tvm -Dtvm-support=enabled
%else
%define enable_tvm -Dtvm-support=disabled
%endif

# Support trix-engine
%if 0%{?trix_engine_support}
%define enable_trix_engine -Dtrix-engine-support=enabled
%else
%define enable_trix_engine -Dtrix-engine-support=disabled
%endif

# Framework priority for each file extension
%define fw_priority_bin ''
%define fw_priority_nb ''

%if "%{?profile}" == "tv"
%define fw_priority_bin 'vd_aifw'
%define fw_priority_nb 'vd_aifw'
%else
%if 0%{openvino_support}
%define fw_priority_bin 'openvino'
%endif
%endif

%define fp16_enabled 0
%ifarch %arm aarch64
%define fp16_enabled 1
# x64/x86 requires GCC >= 12 for fp16 support.
%endif

## Float16 support
%if 0%{?fp16_enabled}
%define fp16_support -Denable-float16=true
%else
%define fp16_support -Denable-float16=false
%endif

%define fw_priority -Dframework-priority-nb=%{fw_priority_nb} -Dframework-priority-bin=%{fw_priority_bin}

%prep
rm -rf ./build
%setup -q
cp %{SOURCE1001} .

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

meson --buildtype=plain --prefix=%{_prefix} --sysconfdir=%{_sysconfdir} --libdir=%{_lib} \
	--bindir=%{nnstbindir} --includedir=include -Dsubplugindir=%{_prefix}/lib/nnstreamer \
	%{enable_tizen} %{element_restriction} %{fw_priority} -Denable-env-var=false -Denable-symbolic-link=false \
	%{enable_tf_lite} %{enable_tf2_lite} %{enable_tf} %{enable_pytorch} %{enable_caffe2} %{enable_python3} \
	%{enable_nnfw_runtime} %{enable_mvncsdk2} %{enable_openvino} %{enable_armnn} %{enable_edgetpu}  %{enable_vivante} \
	%{enable_flatbuf} %{enable_trix_engine} \
	%{enable_tizen_sensor} %{enable_mqtt} %{enable_lua} %{enable_tvm} %{enable_test} %{enable_test_coverage} %{install_test} \
        %{fp16_support} \
	build

ninja -C build %{?_smp_mflags}

export NNSTREAMER_SOURCE_ROOT_PATH=$(pwd)
export NNSTREAMER_BUILD_ROOT_PATH=$(pwd)/build
export GST_PLUGIN_PATH=${NNSTREAMER_BUILD_ROOT_PATH}/gst:${NNSTREAMER_BUILD_ROOT_PATH}/ext
export NNSTREAMER_CONF=${NNSTREAMER_BUILD_ROOT_PATH}/nnstreamer-test.ini
export NNSTREAMER_FILTERS=${NNSTREAMER_BUILD_ROOT_PATH}/ext/nnstreamer/tensor_filter
export NNSTREAMER_DECODERS=${NNSTREAMER_BUILD_ROOT_PATH}/ext/nnstreamer/tensor_decoder
export NNSTREAMER_CONVERTERS=${NNSTREAMER_BUILD_ROOT_PATH}/ext/nnstreamer/tensor_converter
export NNSTREAMER_TRAINERS=${NNSTREAMER_BUILD_ROOT_PATH}/ext/nnstreamer/tensor_trainer

%define test_script $(pwd)/packaging/run_unittests_binaries.sh

# if it's tizen && non-TV, run unittest even if "unit_test"==0 for build-time sanity checks.
%if ( %{with tizen} && "%{?profile}" != "tv" )
%if 0%{nnfw_support}
    bash %{test_script} ./tests/tizen_nnfw_runtime/unittest_nnfw_runtime_raw
%endif
%if 0%{tizen_sensor_support}
    bash %{test_script} ./tests/tizen_sensor/unittest_tizen_sensor
%endif
%endif #if tizen

# If "unit_test"==0, don't run these for the sake of build speed.
%if 0%{?unit_test}
    bash %{test_script} ./tests
    bash %{test_script} ./tests/cpp_methods
    bash %{test_script} ./tests/nnstreamer_filter_extensions_common
%if 0%{?nnstreamer_edge_support}
    bash %{test_script} ./tests/nnstreamer_edge
%endif
%if 0%{mvncsdk2_support}
    LD_LIBRARY_PATH=${NNSTREAMER_BUILD_ROOT_PATH}/tests/nnstreamer_filter_mvncsdk2:. bash %{test_script} ./tests/nnstreamer_filter_mvncsdk2/unittest_filter_mvncsdk2
%endif
%if 0%{edgetpu_support}
    LD_LIBRARY_PATH=${NNSTREAMER_BUILD_ROOT_PATH}/tests/nnstreamer_filter_edgetpu:. bash %{test_script} ./tests/nnstreamer_filter_edgetpu/unittest_edgetpu
%endif
%ifarch %arm x86_64 aarch64 ## @todo This is a workaround. Need to remove %ifarch/%endif some day.
    bash %{test_script} ./tests/nnstreamer_filter_tvm
%endif
    pushd tests

    %ifarch %arm aarch64
    ## @todo Workaround for QEMU compatibility issue. Newer qemu may be ok with this.
    export SKIP_QEMU_ARM_INCOMPATIBLE_TESTS=1
    %else
    export SKIP_QEMU_ARM_INCOMPATIBLE_TESTS=0
    %endif

%if 0%{?fp16_enabled}
    ## If fp16 tests fail, stop the build!
    export FLOAT16_SUPPORTED=1
%endif
    ssat -n -p=1 --summary summary.txt -cn _n
    popd

python3 tools/development/count_test_cases.py build tests/summary.txt
%else
%if 0%{?edge_test} && 0%{?nnstreamer_edge_support}
    bash %{test_script} tests/nnstreamer_edge
    pushd tests/nnstreamer_edge
    ssat -n -p=1 --summary summary.txt -cn _n
    popd
    python3 tools/development/count_test_cases.py build tests/nnstreamer_edge/summary.txt
%endif
%endif

%install
DESTDIR=%{buildroot} ninja -C build install

mkdir -p %{buildroot}%{_bindir}
pushd %{buildroot}%{_bindir}
ln -sf %{_prefix}/%{nnstbindir}/nnstreamer-check nnstreamer-check
ln -sf %{_prefix}/%{nnstbindir}/nnstreamer-parser nnstreamer-parser
popd

%if 0%{?python3_support}
mkdir -p %{buildroot}%{python3_sitelib}
pushd %{buildroot}%{python3_sitelib}
ln -sf %{_libdir}/nnstreamer_python3.so nnstreamer_python.so
popd
%endif

%if 0%{?testcoverage}
##
# The included directories are:
#
# gst: the nnstreamer elements
# nnstreamer_example: custom plugin examples
# common: common libraries for gst (elements)
# include: common library headers and headers for external code (packaged as "devel")
#
# Intentionally excluded directories are:
#
# tests: We are not going to show testcoverage of the test code itself or example applications
#
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
# Exclude generated files (e.g., Orc, Protobuf) and device-dependent filters.
lcov -r unittest.info "*/*-orc.*" "*/tests/*" "*/tools/*" "*/meson*/*" "*/*@sha/*" "*/*_openvino*" "*/*_edgetpu*" "*/*_movidius_ncsdk2*" "*/*.so.p/*" -o unittest-filtered.info
# Visualize the report
genhtml -o result unittest-filtered.info -t "nnstreamer %{version}-%{release} ${VCS}" --ignore-errors source -p ${RPM_BUILD_DIR}

mkdir -p %{buildroot}%{_datadir}/nnstreamer/unittest/
cp -r result %{buildroot}%{_datadir}/nnstreamer/unittest/
%endif  # test coverage

%post -p /sbin/ldconfig

%postun -p /sbin/ldconfig

%files

%files core
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
%license LICENSE
%{_prefix}/lib/nnstreamer/decoders/libnnstreamer_decoder_bounding_boxes.so
%{_prefix}/lib/nnstreamer/decoders/libnnstreamer_decoder_pose_estimation.so
%{_prefix}/lib/nnstreamer/decoders/libnnstreamer_decoder_image_segment.so
%{_prefix}/lib/nnstreamer/decoders/libnnstreamer_decoder_image_labeling.so
%{_prefix}/lib/nnstreamer/decoders/libnnstreamer_decoder_direct_video.so
%{_prefix}/lib/nnstreamer/decoders/libnnstreamer_decoder_octet_stream.so
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_cpp.so
%{gstlibdir}/libnnstreamer.so
%if 0%{?nnstreamer_edge_support}
%{gstlibdir}/libgstedge.so
%endif
%{_libdir}/libnnstreamer.so

%files single
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
%license LICENSE
%{_libdir}/libnnstreamer-single.so

%files default-configuration
%config %{_sysconfdir}/nnstreamer.ini

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
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_tensorflow1-lite.so
%endif

# for tensorflow2-lite
%if 0%{?tensorflow2_lite_support}
%files tensorflow2-lite
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_tensorflow2-lite.so
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_tensorflow2-lite-custom.so
%endif

%if 0%{?python3_support}
%files python3
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
%{_libdir}/nnstreamer_python3.so
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_python3.so
%{_prefix}/lib/nnstreamer/converters/libnnstreamer_converter_python3.so
%{_prefix}/lib/nnstreamer/decoders/libnnstreamer_decoder_python3.so
%{python3_sitelib}/nnstreamer_python.so
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
%{_prefix}/lib/nnstreamer/decoders/libnnstreamer_decoder_flexbuf.so
%{_prefix}/lib/nnstreamer/converters/libnnstreamer_converter_flexbuf.so
%endif

# for pytorch
%if 0%{?pytorch_support}
%files pytorch
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_pytorch.so
%endif

# for caffe2
%if 0%{?caffe2_support}
%files caffe2
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_caffe2.so
%endif

# for lua
%if 0%{?lua_support}
%files lua
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_lua.so
%endif

# for tvm
%if 0%{?tvm_support}
%files tvm
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_tvm.so
%endif

# for snpe
%if 0%{?snpe_support}
# Workaround: Conditionally enable nnstreamer-snpe rpm package
# when existing actual snpe library (snpe.pc)
%files snpe -f ext/nnstreamer/tensor_filter/filter_snpe_list
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
%endif

# for trix-engine
%if 0%{?trix_engine_support}
%files trix-engine
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_trix-engine.so
%endif

%files devel
%{_includedir}/nnstreamer/tensor_if.h
%{_includedir}/nnstreamer/tensor_filter_custom.h
%{_includedir}/nnstreamer/tensor_filter_custom_easy.h
%{_includedir}/nnstreamer/tensor_converter_custom.h
%{_includedir}/nnstreamer/tensor_decoder_custom.h
%{_includedir}/nnstreamer/nnstreamer_plugin_api_decoder.h
%{_includedir}/nnstreamer/nnstreamer_plugin_api_converter.h
%{_includedir}/nnstreamer/nnstreamer_plugin_api.h
%{_includedir}/nnstreamer/nnstreamer_util.h
%{_includedir}/nnstreamer/tensor_filter_cpp.hh
%{_includedir}/nnstreamer/nnstreamer_cppplugin_api_filter.hh
%{_libdir}/pkgconfig/nnstreamer.pc
%{_libdir}/pkgconfig/nnstreamer-cpp.pc

%files devel-internal
%{_includedir}/nnstreamer/nnstreamer_internal.h
%{_includedir}/nnstreamer/tensor_filter_single.h
%{_libdir}/pkgconfig/nnstreamer-internal.pc

%files devel-static
%{_libdir}/*.a
%exclude %{_libdir}/libnnstreamer-single.a

%files single-devel
%{_includedir}/nnstreamer/tensor_typedef.h
%{_includedir}/nnstreamer/nnstreamer_plugin_api_filter.h
%{_includedir}/nnstreamer/nnstreamer_plugin_api_util.h
%{_includedir}/nnstreamer/nnstreamer_version.h
%{_libdir}/pkgconfig/nnstreamer-single.pc

%files single-devel-static
%{_libdir}/libnnstreamer-single.a

%if 0%{?testcoverage}
%files unittest-coverage
%{_datadir}/nnstreamer/unittest/*
%endif

%if 0%{?nnfw_support}
%files nnfw
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_nnfw.so
%endif

%if 0%{?armnn_support}
%files armnn
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_armnn.so
%endif

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

%if 0%{?grpc_support}
%files grpc
%defattr(-,root,root,-)
%manifest nnstreamer.manifest
%license LICENSE
%{gstlibdir}/libnnstreamer-grpc.so

%if 0%{?protobuf_support}
%files grpc-protobuf
%defattr(-,root,root,-)
%manifest nnstreamer.manifest
%license LICENSE
%{_libdir}/libnnstreamer_grpc_protobuf.so
%endif

%if 0%{?flatbuf_support}
%files grpc-flatbuf
%defattr(-,root,root,-)
%manifest nnstreamer.manifest
%license LICENSE
%{_libdir}/libnnstreamer_grpc_flatbuf.so
%endif

%endif  # grpc_support

%if 0%{?edgetpu_support}
%files edgetpu
%manifest nnstreamer.manifest
%license LICENSE
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_edgetpu.so
%endif

%if 0%{?release_test}
%files unittests
%manifest nnstreamer.manifest
%{_libdir}/libnnstreamer_unittest_util.so
%{_libdir}/libcppfilter_test.so
%if 0%{?mvncsdk2_support}
%{_libdir}/libmvncsdk_test.so
%endif
%{_prefix}/lib/nnstreamer/customfilters/*.so
%{_prefix}/%{nnstbindir}/unittest-nnstreamer/nnstreamer-test.ini
%{_prefix}/%{nnstbindir}/unittest-nnstreamer/tests
%endif

%if 0%{?openvino_support}
%files openvino
%manifest nnstreamer.manifest
%license LICENSE
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_openvino.so
%endif

%files util
%{_bindir}/nnstreamer-check
%{_bindir}/nnstreamer-parser
%{_prefix}/%{nnstbindir}/nnstreamer-check
%{_prefix}/%{nnstbindir}/nnstreamer-parser

%files misc
%{gstlibdir}/libgstjoin.so
%if 0%{?mqtt_support}
%{gstlibdir}/libgstmqtt.so
%endif

%if 0%{?release_test}
%files test-devel
%{_prefix}/%{nnstbindir}/unittest-nnstreamer/subplugin_unittest_template.cc.in
%{_prefix}/%{nnstbindir}/unittest-nnstreamer/nnstreamer-test.ini.in
%endif

%changelog
* Tue Sep 27 2022 MyungJoo Ham <myungjoo.ham@samsung.com>
- Start development of 2.3.0 (2.4.0-RC1) for experimental and unstable features.

* Thu Sep 22 2022 MyungJoo Ham <myungjoo.ham@samsung.com>
- Release of 2.2.0, the new LTS version of 2022. (Tizen 7.0 M2)

* Wed Apr 13 2022 MyungJoo Ham <myungjoo.ham@samsung.com>
- Start development of 2.1.1 (2.2.0-RC2)

* Tue Sep 28 2021 MyungJoo Ham <myungjoo.ham@samsung.com>
- Start development of 2.1.0 (2.2.0-RC1)

* Tue Sep 28 2021 MyungJoo Ham <myungjoo.ham@samsung.com>
- Release of 2.0.0, the new LTS version of 2021.

* Thu Jun 03 2021 MyungJoo Ham <myungjoo.ham@samsung.com>
- Start development of 1.7.2 (1.8.0-RC3)

* Fri Nov 20 2020 MyungJoo Ham <myungjoo.ham@samsung.com>
- Start development of 1.7.1 (1.8.0-RC2)

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
