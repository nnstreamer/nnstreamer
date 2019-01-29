
if have_tensorflow
  tensor_filter_tfcore_lib_shared = shared_library('nnstreamer_filter_tensorflow',
      [
        'tensor_filter_tensorflow_core.cc',
        'tensor_filter_tensorflow.c'
      ],
    cpp_args: ['-Wno-sign-compare'],
    dependencies: [tf_dep, protobuf_dep, glib_dep, gst_dep, orc_dep, nnstreamer_dep],
    include_directories: nnstreamer_inc,
    install: true,
    install_dir: subplugins_install_dir
  )

  tensor_filter_tfcore_lib_static = static_library('nnstreamer_filter_tensorflow',
      [
        'tensor_filter_tensorflow_core.cc',
        'tensor_filter_tensorflow.c'
      ],
    cpp_args: ['-Wno-sign-compare'],
    dependencies: [tf_dep, protobuf_dep, glib_dep, gst_dep, orc_dep, nnstreamer_dep],
    include_directories: nnstreamer_inc,
    install: true,
    install_dir: subplugins_install_dir
  )

  tensor_filter_tfcore_lib = tensor_filter_tfcore_lib_shared
  if get_option('default_library') == 'static'
    tensor_filter_tfcore_lib = tensor_filter_tfcore_lib_static
  endif

  tensor_filter_tfcore_dep = declare_dependency(link_with: tensor_filter_tfcore_lib)
  nnstreamer_extra_deps += tensor_filter_tfcore_dep
endif

if have_tensorflow_lite
  tensor_filter_tflitecore_lib_shared = shared_library('nnstreamer_filter_tensorflow-lite',
      [
        'tensor_filter_tensorflow_lite_core.cc',
        'tensor_filter_tensorflow_lite.c'
      ],
    cpp_args: ['-Wno-sign-compare'],
    dependencies: [tflite_dep, thread_dep, libdl_dep, glib_dep, gst_dep, orc_dep, nnstreamer_dep],
    include_directories: nnstreamer_inc,
    install: true,
    install_dir: subplugins_install_dir
  )

  tensor_filter_tflitecore_lib_static = static_library('nnstreamer_filter_tensorflow-lite',
      [
        'tensor_filter_tensorflow_lite_core.cc',
        'tensor_filter_tensorflow_lite.c'
      ],
    cpp_args: ['-Wno-sign-compare'],
    dependencies: [tflite_dep, thread_dep, libdl_dep, glib_dep, gst_dep, orc_dep, nnstreamer_dep],
    include_directories: nnstreamer_inc,
    install: true,
    install_dir: subplugins_install_dir
  )

  tensor_filter_tflitecore_lib = tensor_filter_tflitecore_lib_shared
  if get_option('default_library') == 'static'
    tensor_filter_tflitecore_lib = tensor_filter_tflitecore_lib_static
  endif

  tensor_filter_tflitecore_dep = declare_dependency(link_with: tensor_filter_tflitecore_lib)
  nnstreamer_extra_deps += tensor_filter_tflitecore_dep
endif
