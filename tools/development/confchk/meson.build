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
