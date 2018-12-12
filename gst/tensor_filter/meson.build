tensor_filter_sources = [
  'tensor_filter.c',
  'tensor_filter_custom.c'
]

if have_tensorflow
  tensor_filter_sources += ['tensor_filter_tensorflow.c']

  tensor_filter_tfcore_lib = library('tensor_filter_tfcore',
    ['tensor_filter_tensorflow_core.cc'],
    cpp_args: nnstreamer_base_args + ['-Wno-sign-compare'],
    dependencies: [tf_dep, protobuf_dep, glib_dep, gst_dep],
    include_directories: nnstreamer_inc,
    install: true,
    install_dir: subplugins_install_dir
  )

  tensor_filter_tfcore_dep = declare_dependency(link_with: tensor_filter_tfcore_lib)
  nnstreamer_extra_deps += tensor_filter_tfcore_dep
endif

if have_tensorflow_lite
  tensor_filter_sources += ['tensor_filter_tensorflow_lite.c']

  tensor_filter_tflitecore_lib = library('tensor_filter_tflitecore',
    ['tensor_filter_tensorflow_lite_core.cc'],
    cpp_args: nnstreamer_base_args + ['-Wno-sign-compare'],
    dependencies: [tflite_dep, libpthread_dep, libdl_dep, glib_dep, gst_dep],
    include_directories: nnstreamer_inc,
    install: true,
    install_dir: subplugins_install_dir
  )

  tensor_filter_tflitecore_dep = declare_dependency(link_with: tensor_filter_tflitecore_lib)
  nnstreamer_extra_deps += tensor_filter_tflitecore_dep
endif

foreach s : tensor_filter_sources
  nnstreamer_sources += join_paths(meson.current_source_dir(), s)
endforeach
