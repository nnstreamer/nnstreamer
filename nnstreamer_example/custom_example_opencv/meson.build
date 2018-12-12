opencv_dep = dependency('opencv')

nnstreamer_customfilter_opencv_scaler = shared_library('nnstreamer_customfilter_opencv_scaler',
  'nnstreamer_customfilter_opencv_scaler.cc',
  dependencies: [glib_dep, opencv_dep],
  include_directories: nnstreamer_inc,
  install: get_option('INSTALL_EXAMPLES'),
  install_dir: subplugins_install_dir
)

nnstreamer_customfilter_opencv_average = shared_library('nnstreamer_customfilter_opencv_average',
  'nnstreamer_customfilter_opencv_average.cc',
  dependencies: [glib_dep, opencv_dep],
  include_directories: nnstreamer_inc,
  install: get_option('INSTALL_EXAMPLES'),
  install_dir: subplugins_install_dir
)
