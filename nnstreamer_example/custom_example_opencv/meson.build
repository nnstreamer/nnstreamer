opencv_dep = dependency('opencv')

library('nnstreamer_customfilter_opencv_scaler',
  'nnstreamer_customfilter_opencv_scaler.cc',
  dependencies: [glib_dep, opencv_dep],
  include_directories: nnstreamer_inc,
  install: get_option('install-example'),
  install_dir: customfilter_install_dir
)

library('nnstreamer_customfilter_opencv_average',
  'nnstreamer_customfilter_opencv_average.cc',
  dependencies: [glib_dep, opencv_dep],
  include_directories: nnstreamer_inc,
  install: get_option('install-example'),
  install_dir: customfilter_install_dir
)
