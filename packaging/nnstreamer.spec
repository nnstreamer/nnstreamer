Name:		nnstreamer
Summary:	gstremaer plugins for neural networks
Version:	0.0.1
Release:	1
Group:		Applications/Multimedia
Packager:	MyungJoo Ham <myungjoo.ham@samsung.com>
License:	LGPL-2.0
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

%description
NNStreamer is a set of gstreamer plugins to support general neural networks
and their plugins in a gstreamer stream.

%prep
%setup -q

%build

mkdir -p build
pushd build
%cmake .. -DTIZEN=ON
make %{?_smp_mflags}
popd

# DO THE TEST!

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
%{_libdir}/*

%changelog
* Fri May 25 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
- Packaged tensor_convert plugin.
