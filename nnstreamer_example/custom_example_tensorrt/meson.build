if tensorrt_support_is_available
  library('nnstreamer_customfilter_tensorrt_reshape',
    'nnstreamer_customfilter_example_tensorrt_reshape.cc',
    dependencies: [glib_dep, gst_dep, nnstreamer_dep, tensorrt_deps],
    install: get_option('install-test'),
    install_dir: customfilter_install_dir
  )
endif
