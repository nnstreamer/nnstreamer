Name:		nnstreamer
Summary:	gstremaer plugins for neural networks
Version:	0.0.1
Release:	1
Group:		Applications/Multimedia
Packager:	MyungJoo Ham <myungjoo.ham@samsung.com>
License:	LGPL-2.1+ and Apache-2.0
Source0:	nnstreamer-%{version}.tar.gz
Source1001:	nnstreamer.manifest
Source2001:	testcase_tensor_converter.tar.gz

Requires:	gstreamer >= 1.8.0
Requires:	libdlog
BuildRequires:	gstreamer-devel
BuildRequires:	gst-plugins-base-devel
BuildRequires:	glib2-devel
BuildRequires:	cmake
BuildRequires:	libdlog-devel

# To run test cases, we need gst plugins
BuildRequires:	gst-plugins-good
BuildRequires:	gst-plugins-good-extra
BuildRequires:	gst-plugins-base
# and gtest
BuildRequires:	gtest-devel

%description
NNStreamer is a set of gstreamer plugins to support general neural networks
and their plugins in a gstreamer stream.

%prep
%setup -q
cp %{SOURCE1001} .

%build

mkdir -p build
pushd build
%cmake .. -DTIZEN=ON
make %{?_smp_mflags}
popd

# DO THE TEST!

pushd build
./unittest_common
popd

pushd tensor_converter/test
# We skip testcase gen because it requires PIL, which requires tk.
# Use the pre-generated test cases
tar -xf %{SOURCE2001}
./runTest.sh -skipgen
popd

%install
pushd build
%make_install
popd

%files
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
# The libraries are in LGPLv2.1 (testcases and non GST-plugin components are APL2)
%license LICENSE.LGPLv2.1
%{_libdir}/*

%changelog
* Fri May 25 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
- Packaged tensor_convert plugin.
