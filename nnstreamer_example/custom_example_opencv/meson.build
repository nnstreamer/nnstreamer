opencv_dep = dependency('opencv', required: false)

if opencv_dep.found()
  library('nnstreamer_customfilter_opencv_scaler',
    'nnstreamer_customfilter_opencv_scaler.cc',
    dependencies: [glib_dep, gst_dep, nnstreamer_dep, opencv_dep],
    install: get_option('install-test'),
    install_dir: customfilter_install_dir
  )

  library('nnstreamer_customfilter_opencv_average',
    'nnstreamer_customfilter_opencv_average.cc',
    dependencies: [glib_dep, gst_dep, nnstreamer_dep, opencv_dep],
    install: get_option('install-test'),
    install_dir: customfilter_install_dir
  )
endif
