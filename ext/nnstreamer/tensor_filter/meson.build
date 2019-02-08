if have_tensorflow
  nnstreamer_filter_tf_sources = [
    'tensor_filter_tensorflow.c',
    'tensor_filter_tensorflow_core.cc'
  ]

  nnstreamer_filter_tf_deps = [tf_dep, protobuf_dep, glib_dep, gst_dep, nnstreamer_dep]

  shared_library('nnstreamer_filter_tensorflow',
    nnstreamer_filter_tf_sources,
    cpp_args: ['-Wno-sign-compare'],
    dependencies: nnstreamer_filter_tf_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_tensorflow',
    nnstreamer_filter_tf_sources,
    cpp_args: ['-Wno-sign-compare'],
    dependencies: nnstreamer_filter_tf_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if have_tensorflow_lite
  nnstreamer_filter_tflite_sources = [
    'tensor_filter_tensorflow_lite.c',
    'tensor_filter_tensorflow_lite_core.cc'
  ]

  nnstreamer_filter_tflite_deps = [tflite_dep, thread_dep, libdl_dep, glib_dep, gst_dep, nnstreamer_dep]

  shared_library('nnstreamer_filter_tensorflow-lite',
    nnstreamer_filter_tflite_sources,
    cpp_args: ['-Wno-sign-compare'],
    dependencies: nnstreamer_filter_tflite_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_tensorflow-lite',
    nnstreamer_filter_tflite_sources,
    cpp_args: ['-Wno-sign-compare'],
    dependencies: nnstreamer_filter_tflite_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif
