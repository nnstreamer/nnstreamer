project('nnstreamer-check', 'c',
  version: '1.0.0',
  license: ['LGPL-2.1'],
  meson_version: '>=0.50.0',
  default_options: [
    'werror=true',
    'warning_level=1',
    'c_std=gnu89',
  ]
)

cc = meson.get_compiler('c')
gst_api_verision = '1.0'
glib_dep = dependency('glib-2.0')
gst_dep = dependency('gstreamer-' + gst_api_verision)
gst_base_dep = dependency('gstreamer-base-' + gst_api_verision)
libdl_dep = cc.find_library('dl') # DL library

nnstchk_deps = [
  glib_dep,
  gst_dep,
  gst_base_dep,
  libdl_dep,
]

nnstchk_exec = executable('nnstreamer-check',
  'confchk.c',
  dependencies: nnstchk_deps,
  include_directories: ['../../../gst/nnstreamer', '../../../gst/nnstreamer/include'],
  install: true,
  install_dir: join_paths(get_option('prefix'), get_option('bindir')),
)
