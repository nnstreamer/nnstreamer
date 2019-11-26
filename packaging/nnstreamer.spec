# Execute gbs with --define "testcoverage 1" in case that you must get unittest coverage statistics
%define		gstpostfix	gstreamer-1.0
%define		gstlibdir	%{_libdir}/%{gstpostfix}
%define		nnstexampledir	/usr/lib/nnstreamer/bin
%define		tensorflow_support	0
%define		enable_nnfw	1

# If it is tizen, we can export Tizen API packages.
%bcond_with tizen

Name:		nnstreamer
Summary:	gstreamer plugins for neural networks
# Synchronize the version information among Ubuntu, Tizen, Android, and Meson.
# 1. Ubuntu : ./debian/changelog
# 2. Tizen  : ./packaging/nnstreamer.spec
# 3. Android: ./jni/nnstreamer.mk
# 4. Meson  : ./meson.build
Version:	1.0.0
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


Requires:	gstreamer >= 1.8.0
BuildRequires:	gstreamer-devel
BuildRequires:	gst-plugins-base-devel
BuildRequires:	glib2-devel
BuildRequires:	meson >= 0.50.0

# To run test cases, we need gst plugins
BuildRequires:	gst-plugins-good
BuildRequires:	gst-plugins-good-extra
BuildRequires:	gst-plugins-base
# and gtest
BuildRequires:	gtest-devel
# a few test cases uses python
BuildRequires:	python
BuildRequires:	python-numpy
# for python custom filters
BuildRequires:	pkgconfig(python2)
BuildRequires:	python-numpy-devel
# Testcase requires bmp2png, which requires libpng
BuildRequires:  pkgconfig(libpng)
# for tensorflow-lite
BuildRequires: tensorflow-lite-devel
# custom_example_opencv filter requires opencv-devel
BuildRequires: opencv-devel
# For './testAll.sh' time limit.
BuildRequires: procps
# for tensorflow
%if 0%{?tensorflow_support}
BuildRequires: protobuf-devel >= 3.4.0
BuildRequires: tensorflow
BuildRequires: tensorflow-devel
%endif

%if 0%{?testcoverage}
BuildRequires: lcov
# BuildRequires:	taos-ci-unittest-coverage-assessment
%endif

%define		enable_nnfw_runtime	-Denable-nnfw-runtime=false
%define		enable_nnfw_r	0
%if %{with tizen}
BuildRequires:	pkgconfig(dpm)
BuildRequires:	pkgconfig(mm-resource-manager)
BuildRequires:	pkgconfig(mm-camcorder)
BuildRequires:	pkgconfig(capi-privacy-privilege-manager)
BuildRequires:	pkgconfig(capi-system-info)
BuildRequires:	pkgconfig(capi-base-common)
BuildRequires:	pkgconfig(dlog)
BuildRequires:	pkgconfig(libmvnc)
BuildRequires:	gst-plugins-bad-devel
BuildRequires:	gst-plugins-base-devel

%if 0%{?enable_nnfw}
%ifarch %arm aarch64
# Tizen 5.5 M2+ support nn-runtime (nnfw)
# As of 2019-09-24, unfortunately, nnfw does not support pkg-config
BuildRequires:  nnfw-devel
BuildRequires:  libarmcl
BuildConflicts: libarmcl-release
%define		enable_nnfw_runtime	-Denable-nnfw-runtime=true
%define		enable_nnfw_r	1
%endif
%endif
%endif

# Unit Testing Uses SSAT (hhtps://github.com/myungjoo/SSAT.git)
%if 0%{?unit_test}
BuildRequires: ssat
%endif

# For ORC (Oil Runtime Compiler)
BuildRequires:	pkgconfig(orc-0.4)

%if %{with tizen}
BuildRequires:	pkgconfig(sensor)
BuildRequires:	capi-system-sensor-devel
%endif

# Note that debug packages generate an additional build and storage cost.
# If you do not need debug packages, run '$ gbs build ... --define "_skip_debug_rpm 1"'.

%if "%{?_skip_debug_rpm}" == "1"
%global debug_package %{nil}
%global __debug_install_post %{nil}
%endif

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

%package tensorflow-lite
Summary:	NNStreamer TensorFlow Lite Support
Requires:	nnstreamer = %{version}-%{release}
# tensorflow-lite provides .a file and it's embedded into the subplugin. No dep to tflite.
%description tensorflow-lite
NNStreamer's tensor_fliter subplugin of TensorFlow Lite.

%package nnfw
Summary:	NNStreamer Tizen-nnfw runtime support
Requires:	nnfw
%description nnfw
NNStreamer's tensor_filter subplugin of Tizen-NNFW Runtime. (5.5 M2 +)

%package -n nnstreamer-python2
Summary:  NNStreamer Python Custom Filter Support
Requires: nnstreamer = %{version}-%{release}
Requires: python
%description -n nnstreamer-python2
NNStreamer's tensor_filter subplugin of Python (2.7).

%package devel
Summary:	Development package for custom tensor operator developers (tensor_filter/custom)
Requires:	nnstreamer = %{version}-%{release}
Requires:	glib2-devel
Requires:	gstreamer-devel
%description devel
Development package for custom tensor operator developers (tensor_filter/custom).
This contains corresponding header files and .pc pkgconfig file.

%package custom-filter-example
Summary:	NNStreamer example custom plugins and test plugins
Requires:	nnstreamer = %{version}-%{release}
%description custom-filter-example
Example custom tensor_filter subplugins and
plugins created for test purpose.

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
%description -n capi-nnstreamer
Tizen Native API wrapper for NNStreamer.
You can construct a data stream pipeline with neural networks easily.

%post -n capi-nnstreamer -p /sbin/ldconfig
%postun -n capi-nnstreamer -p /sbin/ldconfig

%package -n capi-nnstreamer-devel
Summary:	Tizen Native API Devel Kit for NNStreamer
Group:		Multimedia/Framework
Requires:	capi-nnstreamer = %{version}-%{release}
%description -n capi-nnstreamer-devel
Developmental kit for Tizen Native NNStreamer API.

%package -n nnstreamer-tizen-internal-capi-devel
Summary:	Tizen internal API to construct the pipeline
Group:		Multimedia/Framework
Requires:	capi-nnstreamer-devel = %{version}-%{release}
%description -n nnstreamer-tizen-internal-capi-devel
Tizen internal API to construct the pipeline without the permissions.

%package	ncsdk2
Summary:	NNStreamer Intel Movidius NCSDK2 support
Group:		Multimedia/Framework
%description	ncsdk2
NNStreamer's tensor_fliter subplugin of Intel Movidius Neural Compute stick SDK2.
%endif

%package cpp
Summary:	NNStreamer Custom Plugin Support for C++ Classes
Requires:	nnstreamer = %{version}-%{release}
%description cpp
With this package, you may use C++ classes as yet another tensor-filter subplugins of nnstreamer pipelines.

%post cpp -p /sbin/ldconfig
%postun cpp -p /sbin/ldconfig

%package cpp-devel
Summary:	NNStreamer Custom Plugin Development Support for C++ Classes
Requires:	nnstreamer-cpp = %{version}-%{release}
%description cpp-devel
With this package, you may write C++ classes as yet another tensor-filter subplugins of nnstreamer pipelines.
Note that there is no .pc file for this package because nnstreamer.pc file may be used for developing this.

# Define build options
%if %{with tizen}
%define enable_tizen -Denable-tizen=true -Denable-tizen-sensor=true
%define enable_api -Denable-capi=true
%define enable_mvncsdk2 -Denable-movidius-ncsdk2=true

# Add Tizen's sensor framework API integration
%package tizen-sensor
Summary:	NNStreamer integration of tizen sensor framework (tensor_src_tizensensor)
Requires:	nnstreamer = %{version}-%{release}
Requires:	capi-system-sensor
%description tizen-sensor
You can include Tizen sensor framework nodes as source elements of GStreamer/NNStreamer pipelines with this package.

# Element restriction in Tizen
%define restricted_element	'capsfilter input-selector output-selector queue tee valve appsink appsrc audioconvert audiorate audioresample audiomixer videoconvert videocrop videorate videoscale videoflip videomixer compositor fakesrc fakesink filesrc filesink audiotestsrc videotestsrc jpegparse jpegenc jpegdec pngenc pngdec tcpclientsink tcpclientsrc tcpserversink tcpserversrc udpsink udpsrc xvimagesink ximagesink evasimagesink evaspixmapsink glimagesink theoraenc lame vorbisenc wavenc volume oggmux avimux matroskamux v4l2src avsysvideosrc camerasrc tvcamerasrc pulsesrc fimcconvert'

%define restriction -Denable-element-restriction=true -Drestricted-elements=%{restricted_element}
%else
%define enable_tizen -Denable-tizen=false
%define enable_api -Denable-capi=false
%define restriction -Denable-element-restriction=false
%endif

# Support tensorflow
%if 0%{?tensorflow_support}
%define enable_tf -Denable-tensorflow=true
%else
%define enable_tf -Denable-tensorflow=false
%endif

%prep
%setup -q
cp %{SOURCE1001} .
%if %{with tizen}
pushd %{_sourcedir}
sh %{SOURCE1} %{name} %{version}
popd
cp %{SOURCE1002} .
rm -rf ./api/capi/include/platform
%endif
cp %{SOURCE1001} ./nnstreamer-cpp.manifest

%build
# Remove compiler flags for meson to decide the cpp version
CXXFLAGS=`echo $CXXFLAGS | sed -e "s|-std=gnu++11||"`

%if 0%{?testcoverage}
CXXFLAGS="${CXXFLAGS} -fprofile-arcs -ftest-coverage"
CFLAGS="${CFLAGS} -fprofile-arcs -ftest-coverage"
%endif

mkdir -p build

meson --buildtype=plain --prefix=%{_prefix} --sysconfdir=%{_sysconfdir} --libdir=%{_libdir} \
	--bindir=%{nnstexampledir} --includedir=%{_includedir} -Dinstall-example=true %{enable_tf} \
	-Denable-pytorch=false -Denable-caffe2=false -Denable-env-var=false -Denable-symbolic-link=false \
	%{enable_api} %{enable_tizen} %{restriction} %{enable_nnfw_runtime} %{enable_mvncsdk2} build

ninja -C build %{?_smp_mflags}

%if 0%{?unit_test}
    export NNSTREAMER_BUILD_ROOT_PATH=$(pwd)
    pushd build
    export GST_PLUGIN_PATH=$(pwd)/gst/nnstreamer
    export NNSTREAMER_CONF=$(pwd)/nnstreamer-test.ini
    export NNSTREAMER_FILTERS=$(pwd)/ext/nnstreamer/tensor_filter
    export NNSTREAMER_DECODERS=$(pwd)/ext/nnstreamer/tensor_decoder
    ./tests/unittest_common --gtest_output="xml:unittest_common.xml"
    ./tests/unittest_sink --gst-plugin-path=. --gtest_output="xml:unittest_sink.xml"
    ./tests/unittest_plugins --gst-plugin-path=. --gtest_output="xml:unittest_plugins.xml"
    ./tests/unittest_src_iio --gst-plugin-path=. --gtest_output="xml:unittest_src_iio.xml"
    ./tests/tizen_capi/unittest_tizen_capi --gst-plugin-path=. --gtest_output="xml:unittest_tizen_capi.xml"
%if 0%{?enable_nnfw_r}
    ./tests/tizen_nnfw_runtime/unittest_nnfw_runtime_raw --gst-plugin-path=. --gtest_output="xml:unittest_nnfw_runtime_raw.xml"
%endif
%if %{with tizen}
    ln -s ext/nnstreamer/tensor_source/*.so .
    ./tests/tizen_capi/unittest_tizen_sensor --gst-plugin-path=. --gtest_output="xml:unittest_tizen_sensor.xml"
%endif
    popd
    pushd tests
    ssat -n
    popd
%endif

%install
DESTDIR=%{buildroot} ninja -C build %{?_smp_mflags} install

pushd %{buildroot}%{_libdir}
ln -sf %{gstlibdir}/libnnstreamer.so libnnstreamer.so
popd

mkdir -p %{buildroot}%{python_sitelib}
pushd %{buildroot}%{python_sitelib}
ln -sf %{_prefix}/lib/nnstreamer/filters/nnstreamer_python2.so nnstreamer_python.so
popd

# Hotfix: Support backward compatibility
pushd %{buildroot}%{_libdir}
ln -sf ./libcapi-nnstreamer.so libcapi-nnstreamer.so.0
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
    $(pwd)/tests/unittestcoverage.py module $(pwd)/gst $(pwd)/ext %testtarget

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
    # Exclude generated files (Orc)
    lcov -r unittest.info "*/*-orc.*" "*/tests/*" "*/meson*/*" -o unittest-filtered.info
    # Visualize the report
    genhtml -o result unittest-filtered.info -t "nnstreamer %{version}-%{release} ${VCS}" --ignore-errors source -p ${RPM_BUILD_DIR}
%endif

%if 0%{?testcoverage}
mkdir -p %{buildroot}%{_datadir}/nnstreamer/unittest/
cp -r result %{buildroot}%{_datadir}/nnstreamer/unittest/
%endif

%post -p /sbin/ldconfig

%postun -p /sbin/ldconfig

%files
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
%license LICENSE
%{_prefix}/lib/nnstreamer/decoders/libnnstreamer_decoder_*.so
%{gstlibdir}/libnnstreamer.so
%{_libdir}/libnnstreamer.so
%{_sysconfdir}/nnstreamer.ini

# for tensorflow
%if 0%{?tensorflow_support}
%files tensorflow
%manifest capi-nnstreamer.manifest
%defattr(-,root,root,-)
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_tensorflow.so
%endif

%files tensorflow-lite
%manifest capi-nnstreamer.manifest
%defattr(-,root,root,-)
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_tensorflow-lite.so

%if 0%{?enable_nnfw_r}
%files nnfw
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_nnfw.so
%endif

%files -n nnstreamer-python2
%manifest capi-nnstreamer.manifest
%defattr(-,root,root,-)
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_python2.so
%{_prefix}/lib/nnstreamer/filters/nnstreamer_python2.so
%{python_sitelib}/nnstreamer_python.so

%files devel
%{_includedir}/nnstreamer/tensor_typedef.h
%{_includedir}/nnstreamer/tensor_filter_custom.h
%{_includedir}/nnstreamer/nnstreamer_plugin_api_filter.h
%{_includedir}/nnstreamer/nnstreamer_plugin_api_decoder.h
%{_includedir}/nnstreamer/nnstreamer_plugin_api.h
%{_libdir}/*.a
%exclude %{_libdir}/libcapi*.a
%{_libdir}/pkgconfig/nnstreamer.pc

%if 0%{?testcoverage}
%files unittest-coverage
%{_datadir}/nnstreamer/unittest/*
%endif

%files custom-filter-example
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
%license LICENSE
%{_prefix}/lib/nnstreamer/customfilters/*.so

%if %{with tizen}
%files -n capi-nnstreamer
%manifest capi-nnstreamer.manifest
%license LICENSE
%{_libdir}/libcapi-nnstreamer.so
%{_libdir}/libcapi-nnstreamer.so.*

%files -n capi-nnstreamer-devel
%{_includedir}/nnstreamer/nnstreamer.h
%{_includedir}/nnstreamer/nnstreamer-single.h
%{_libdir}/pkgconfig/capi-nnstreamer.pc
%{_libdir}/libcapi-nnstreamer.a

%files -n nnstreamer-tizen-internal-capi-devel
%{_includedir}/nnstreamer/nnstreamer-tizen-internal.h

%files -n nnstreamer-ncsdk2
%defattr(-,root,root,-)
%manifest nnstreamer.manifest
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_movidius-ncsdk2.so
%endif

%files cpp
%manifest nnstreamer-cpp.manifest
%license LICENSE
%{_prefix}/lib/nnstreamer/filters/libnnstreamer_filter_cpp.so

%files cpp-devel
%{_includedir}/nnstreamer/tensor_filter_cpp.h

%if %{with tizen}
%files tizen-sensor
%{gstlibdir}/libnnstreamer-tizen-sensor.so
%endif

%changelog
* Thu Sep 26 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
- Release of 1.0.0 (1.0 RC2)

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
