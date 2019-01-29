
if have_tensorflow
  tensor_filter_tfcore_lib_shared = shared_library('nnstreamer_filter_tensorflow',
      [
        'tensor_filter_tensorflow_core.cc',
        'tensor_filter_tensorflow.c'
      ],
    cpp_args: ['-Wno-sign-compare'],
    dependencies: [tf_dep, protobuf_dep, nnstreamer_dep],
    include_directories: nnstreamer_inc,
    install: true,
    install_dir: join_paths(subplugin_install_prefix, 'filters')
  )

  tensor_filter_tfcore_lib_static = static_library('nnstreamer_filter_tensorflow',
      [
        'tensor_filter_tensorflow_core.cc',
        'tensor_filter_tensorflow.c'
      ],
    cpp_args: ['-Wno-sign-compare'],
    dependencies: [tf_dep, protobuf_dep, nnstreamer_dep],
    include_directories: nnstreamer_inc,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if have_tensorflow_lite
  tensor_filter_tflitecore_lib_shared = shared_library('nnstreamer_filter_tensorflow-lite',
      [
        'tensor_filter_tensorflow_lite_core.cc',
        'tensor_filter_tensorflow_lite.c'
      ],
    cpp_args: ['-Wno-sign-compare'],
    dependencies: [tflite_dep, nnstreamer_dep],
    include_directories: nnstreamer_inc,
    install: true,
    install_dir: join_paths(subplugin_install_prefix, 'filters')
  )

  tensor_filter_tflitecore_lib_static = static_library('nnstreamer_filter_tensorflow-lite',
      [
        'tensor_filter_tensorflow_lite_core.cc',
        'tensor_filter_tensorflow_lite.c'
      ],
    cpp_args: ['-Wno-sign-compare'],
    dependencies: [tflite_dep, nnstreamer_dep],
    include_directories: nnstreamer_inc,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif
