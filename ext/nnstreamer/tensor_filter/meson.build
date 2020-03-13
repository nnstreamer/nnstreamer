if get_option('enable-nnfw-runtime')
  filter_sub_nnfw_sources = ['tensor_filter_nnfw.c']

  nnstreamer_filter_nnfw_sources = []
  foreach s : filter_sub_nnfw_sources
    nnstreamer_filter_nnfw_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  nnfw_dep = dependency('nnfw', required: false)
  if not nnfw_dep.found()
    # Until nnfw supports pkg-config, we need to do this primitively.
    nnfw_dep = cc.find_library('nnfw-dev')
  endif
  json_glib_dep = dependency('json-glib-1.0', required: true)
  nnstreamer_filter_nnfw_deps = [glib_dep, gst_dep, nnstreamer_dep, json_glib_dep, nnfw_dep]

  nnfw_plugin_lib = shared_library('nnstreamer_filter_nnfw',
    nnstreamer_filter_nnfw_sources,
    dependencies: nnstreamer_filter_nnfw_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )
  static_library('nnstreamer_filter_nnfw',
    nnstreamer_filter_nnfw_sources,
    dependencies: nnstreamer_filter_nnfw_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )

  nnfw_plugin_dep = declare_dependency(link_with: nnfw_plugin_lib,
    dependencies: nnstreamer_filter_nnfw_deps)
endif

if get_option('enable-armnn')
  filter_sub_armnn_sources = [
    'tensor_filter_armnn.cc'
  ]

  nnstreamer_filter_armnn_sources = []
  foreach s : filter_sub_armnn_sources
    nnstreamer_filter_armnn_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  nnstreamer_filter_armnn_deps = [glib_dep, gst_dep, nnstreamer_dep, armnn_dep]

  armnn_plugin_lib = shared_library('nnstreamer_filter_armnn',
    nnstreamer_filter_armnn_sources,
    dependencies: nnstreamer_filter_armnn_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )
  static_library('nnstreamer_filter_armnn',
    nnstreamer_filter_armnn_sources,
    dependencies: nnstreamer_filter_armnn_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )

  armnn_plugin_dep = declare_dependency(link_with: armnn_plugin_lib,
    dependencies: nnstreamer_filter_armnn_deps)
endif

if get_option('enable-tensorflow')
  filter_sub_tf_sources = ['tensor_filter_tensorflow.cc']

  nnstreamer_filter_tf_sources = []
  foreach s : filter_sub_tf_sources
    nnstreamer_filter_tf_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  nnstreamer_filter_tf_deps = [tf_dep, protobuf_dep, glib_dep, gst_dep, nnstreamer_dep]

  shared_library('nnstreamer_filter_tensorflow',
    nnstreamer_filter_tf_sources,
    dependencies: nnstreamer_filter_tf_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_tensorflow',
    nnstreamer_filter_tf_sources,
    dependencies: nnstreamer_filter_tf_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if get_option('enable-tensorflow-lite')
  filter_sub_tflite_sources = ['tensor_filter_tensorflow_lite.cc']

  nnstreamer_filter_tflite_sources = []
  foreach s : filter_sub_tflite_sources
    nnstreamer_filter_tflite_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  nnstreamer_filter_tflite_deps = [tflite_dep, thread_dep, libdl_dep, glib_dep, gst_dep, nnstreamer_dep]

  if get_option('enable-tflite-nnapi-delegation')
    nnfw_nnapi_lib = dependency('nnfw', required: true)
    if nnfw_nnapi_lib.found()
      nnstreamer_filter_tflite_deps += nnfw_nnapi_lib
    else
      error('Cannot find nnfw')
    endif
  endif

  TFLite_version = tflite_dep.version().split('.')
  TFLite_major = TFLite_version[0].to_int()
  TFLite_minor = TFLite_version[1].to_int()

  tflite_compile_args = []
  if TFLite_major == 1
    if TFLite_minor >= 10
      tflite_compile_args += '-DTFLITE_INT16=1'
      tflite_compile_args += '-DTFLITE_COMPLEX64=1'
      message('data type [INT16] is added for TFLite')
      message('data type [COMPLEX64] is added for TFLite')
    endif

    if TFLite_minor >= 13
      tflite_compile_args += '-DTFLITE_INT8=1'
      message('data type [INT8] is added for TFLite')
    endif

    if TFLite_minor >= 14
      tflite_compile_args += '-DTFLITE_FLOAT16=1'
      message('data type [FLOAT16] is added for TFLite')
    endif
  endif

  tflite_extra_dep = declare_dependency(
    compile_args : tflite_compile_args,
  )

  nnstreamer_filter_tflite_deps += tflite_extra_dep

  shared_library('nnstreamer_filter_tensorflow-lite',
    nnstreamer_filter_tflite_sources,
    dependencies: nnstreamer_filter_tflite_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_tensorflow-lite',
    nnstreamer_filter_tflite_sources,
    dependencies: nnstreamer_filter_tflite_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if get_option('enable-pytorch')
  filter_sub_torch_sources = ['tensor_filter_pytorch.cc']

  nnstreamer_filter_torch_sources = []
  foreach s : filter_sub_torch_sources
    nnstreamer_filter_torch_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  nnstreamer_filter_torch_deps = [torch_dep, glib_dep, gst_dep, nnstreamer_dep]

  shared_library('nnstreamer_filter_pytorch',
    nnstreamer_filter_torch_sources,
    dependencies: nnstreamer_filter_torch_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_pytorch',
    nnstreamer_filter_torch_sources,
    dependencies: nnstreamer_filter_torch_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if get_option('enable-caffe2')
  filter_sub_caffe2_sources = ['tensor_filter_caffe2.cc']

  nnstreamer_filter_caffe2_sources = []
  foreach s : filter_sub_caffe2_sources
    nnstreamer_filter_caffe2_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  nnstreamer_filter_caffe2_deps = [caffe2_dep, protobuf_dep, glib_dep, gst_dep, nnstreamer_dep]

  shared_library('nnstreamer_filter_caffe2',
    nnstreamer_filter_caffe2_sources,
    dependencies: nnstreamer_filter_caffe2_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_caffe2',
    nnstreamer_filter_caffe2_sources,
    dependencies: nnstreamer_filter_caffe2_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

filter_sub_python_sources = ['tensor_filter_python.cc']

nnstreamer_filter_python_sources = []
foreach s : filter_sub_python_sources
  nnstreamer_filter_python_sources += join_paths(meson.current_source_dir(), s)
endforeach

if have_python2
  nnstreamer_filter_python2_deps = [python2_dep, libdl_dep, glib_dep, gst_dep, nnstreamer_dep]

  shared_library('nnstreamer_filter_python2',
    nnstreamer_filter_python_sources,
    dependencies: nnstreamer_filter_python2_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_python2',
    nnstreamer_filter_python_sources,
    dependencies: nnstreamer_filter_python2_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )

  shared_library('nnstreamer_python2',
    'tensor_filter_python_api.c',
    name_prefix: '',
    dependencies: nnstreamer_filter_python2_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )
endif

if have_python3
  nnstreamer_filter_python3_deps = [python3_dep, libdl_dep, glib_dep, gst_dep, nnstreamer_dep]

  shared_library('nnstreamer_filter_python3',
    nnstreamer_filter_python_sources,
    dependencies: nnstreamer_filter_python3_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_python3',
    nnstreamer_filter_python_sources,
    dependencies: nnstreamer_filter_python3_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )

  shared_library('nnstreamer_python3',
    'tensor_filter_python_api.c',
    name_prefix: '',
    dependencies: nnstreamer_filter_python3_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )
endif

if have_movidius_ncsdk2
  filter_sub_mvncsdk2_sources = [
    'tensor_filter_movidius_ncsdk2.c'
  ]

  nnstreamer_filter_mvncsdk2_sources = []
  foreach s : filter_sub_mvncsdk2_sources
    nnstreamer_filter_mvncsdk2_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  nnstreamer_filter_mvncsdk2_deps = [glib_dep, gst_dep, nnstreamer_dep, nns_mvncsdk2_dep]

  shared_library('nnstreamer_filter_movidius-ncsdk2',
    nnstreamer_filter_mvncsdk2_sources,
    dependencies: nnstreamer_filter_mvncsdk2_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_movidius-ncsdk2',
    nnstreamer_filter_mvncsdk2_sources,
    dependencies: nnstreamer_filter_mvncsdk2_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if get_option('enable-cppfilter')
  filter_sub_cpp_sources = ['tensor_filter_cpp.cc']

  nnstreamer_filter_cpp_sources = []
  foreach s : filter_sub_cpp_sources
    nnstreamer_filter_cpp_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  nnstreamer_cpp_lib = shared_library('nnstreamer_filter_cpp',
    nnstreamer_filter_cpp_sources,
    dependencies: [glib_dep, gst_dep, nnstreamer_dep],
    install: true,
    install_dir: filter_subplugin_install_dir
  )
  install_headers(['tensor_filter_cpp.hh'],
    subdir: 'nnstreamer'
  )
  configure_file(input: 'nnstreamer-cpp.pc.in', output: 'nnstreamer-cpp.pc',
    install_dir: join_paths(nnstreamer_libdir, 'pkgconfig'),
    configuration: nnstreamer_install_conf
  )
  nnstreamer_cpp_dep = declare_dependency(link_with: nnstreamer_cpp_lib,
    dependencies: [glib_dep, gst_dep, nnstreamer_dep],
    include_directories: include_directories('.')
  )
endif

if get_option('enable-edgetpu')
  if get_option('enable-tensorflow-lite')
    edgetpu_dep = dependency('edgetpu', required: false)
    if not edgetpu_dep.found()
      # Ubuntu edgetpu package does not have pkgconfig file
      edgetpu_dep = cc.find_library('edgetpu')
    endif
    filter_sub_edgetpu_sources = ['tensor_filter_edgetpu.cc']

    nnstreamer_filter_edgetpu_sources = []
    foreach s : filter_sub_edgetpu_sources
      nnstreamer_filter_edgetpu_sources += join_paths(meson.current_source_dir(), s)
    endforeach

    shared_library('nnstreamer_filter_edgetpu',
      nnstreamer_filter_edgetpu_sources,
      dependencies: [glib_dep, gst_dep, nnstreamer_dep, edgetpu_dep, tflite_dep],
      install: true,
      install_dir: filter_subplugin_install_dir
    )
  else
    error ('enable-tensorflow-lite should be set as \'true\' to build the tensor filter for Edge TPU.')
  endif
endif

if get_option('enable-openvino')
  openvino_deps = []
  openvino_cpp_args = []
  openvino_deps += dependency('openvino', required: true)
  if cxx.has_header('ext_list.hpp')
    openvino_cpuext_lib = cxx.find_library('cpu_extension', required: true)
    if openvino_cpuext_lib.found()
      openvino_cpp_args += '-D__OPENVINO_CPU_EXT__=1'
      openvino_deps += openvino_cpuext_lib
    else
      error('Cannot find the shared object, libcpu_extension.')
    endif
  endif
  filter_sub_openvino_sources = ['tensor_filter_openvino.cc']

  nnstreamer_filter_openvino_sources = []
  foreach s : filter_sub_openvino_sources
    nnstreamer_filter_openvino_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  nnstreamer_filter_openvino_deps = [glib_dep, gst_dep, nnstreamer_dep, openvino_deps]

  nnstreamer_filter_openvino_so_lib = shared_library('nnstreamer_filter_openvino',
    nnstreamer_filter_openvino_sources,
    cpp_args: openvino_cpp_args,
    dependencies: nnstreamer_filter_openvino_deps,
    install: false,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_openvino',
    nnstreamer_filter_openvino_sources,
    cpp_args: openvino_cpp_args,
    dependencies: nnstreamer_filter_openvino_deps,
    install: false,
    install_dir: nnstreamer_libdir
  )

  nnstreamer_filter_openvino_dep = declare_dependency(
    link_with: nnstreamer_filter_openvino_so_lib,
    include_directories: include_directories('.')
  )
endif
