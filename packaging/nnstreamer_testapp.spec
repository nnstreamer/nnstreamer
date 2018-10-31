Name:		nnstreamer_testapp
Summary:	test app for gstremaer plugins for neural networks
Version:	0.0.3
Release:	1rc1
Group:		Applications/Multimedia
Packager:	MyungJoo Ham <myungjoo.ham@samsung.com>
License:	LGPL-2.1
Source0:	nnstreamer_testapp-%{version}.tar.gz
Source1001:	nnstreamer.manifest

Requires:	gstreamer >= 1.8.0
Requires:	nnstreamer
Requires:	gst-plugins-good
Requires:	gst-plugins-good-extra
Requires:	gst-plugins-base
BuildRequires:	pkg-config
BuildRequires:	pkgconfig(nnstreamer)
BuildRequires:	pkgconfig(gstreamer-1.0)
BuildRequires:	pkgconfig(gstreamer-video-1.0)
BuildRequires:	pkgconfig(gstreamer-audio-1.0)
BuildRequires:	pkgconfig(gstreamer-app-1.0)
BuildRequires:	cmake

%description
NNStreamer is a set of gstreamer plugins to support general neural networks
and their plugins in a gstreamer stream.

%prep
%setup -q
cp %{SOURCE1001} .

%build
pushd nnstreamer_example/tizen_app_build_example
mkdir -p build
pushd build
%cmake ..
make %{?_smp_mflags}
popd
popd

%install
pushd nnstreamer_example/tizen_app_build_example
pushd build
%make_install
popd
popd

%files
%manifest nnstreamer.manifest
%defattr(-,root,root,-)
%{_bindir}/*

%changelog
* Wed Jul 18 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
- Tizen Test App Project Started
