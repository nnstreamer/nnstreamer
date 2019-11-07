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
  nnstreamer_filter_nnfw_deps = [glib_dep, gst_dep, nnstreamer_dep, nnfw_dep]


  nnfw_plugin_lib = shared_library('nnstreamer_filter_nnfw',
    nnstreamer_filter_nnfw_sources,
    dependencies: nnstreamer_filter_nnfw_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )
  static_library('nnstreamer_filter_nnfw',
    nnstreamer_filter_nnfw_sources,
    dependencies: nnstreamer_filter_nnfw_deps,
    install: false,
    install_dir: filter_subplugin_install_dir
  )

  nnfw_plugin_dep = declare_dependency(link_with: nnfw_plugin_lib,
    dependencies: nnstreamer_filter_nnfw_deps)
endif

if get_option('enable-tensorflow')
  filter_sub_tf_sources = [
    'tensor_filter_tensorflow.c',
    'tensor_filter_tensorflow_core.cc'
  ]

  nnstreamer_filter_tf_sources = []
  foreach s : filter_sub_tf_sources
    nnstreamer_filter_tf_sources += join_paths(meson.current_source_dir(), s)
  endforeach

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

if get_option('enable-tensorflow-lite')
  filter_sub_tflite_sources = [
    'tensor_filter_tensorflow_lite.c',
    'tensor_filter_tensorflow_lite_core.cc'
  ]

  nnstreamer_filter_tflite_sources = []
  foreach s : filter_sub_tflite_sources
    nnstreamer_filter_tflite_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  nnstreamer_filter_tflite_deps = [tflite_dep, thread_dep, libdl_dep, glib_dep, gst_dep, nnstreamer_dep]

  if get_option('enable-nnfw')
    nnapi_lib = dependency('nnfw', required: true)
    if nnapi_lib.found()
      nnstreamer_filter_tflite_deps += nnapi_lib
    else
      error('Cannot find nnfw')
    endif
  endif

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

if get_option('enable-pytorch')
  filter_sub_torch_sources = [
    'tensor_filter_pytorch.c',
    'tensor_filter_pytorch_core.cc'
  ]

  nnstreamer_filter_torch_sources = []
  foreach s : filter_sub_torch_sources
    nnstreamer_filter_torch_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  nnstreamer_filter_torch_deps = [torch_dep, glib_dep, gst_dep, nnstreamer_dep]

  shared_library('nnstreamer_filter_pytorch',
    nnstreamer_filter_torch_sources,
    cpp_args: ['-Wno-sign-compare'],
    dependencies: nnstreamer_filter_torch_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_pytorch',
    nnstreamer_filter_torch_sources,
    cpp_args: ['-Wno-sign-compare'],
    dependencies: nnstreamer_filter_torch_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if get_option('enable-caffe2')
  filter_sub_caffe2_sources = [
    'tensor_filter_caffe2.c',
    'tensor_filter_caffe2_core.cc'
  ]

  nnstreamer_filter_caffe2_sources = []
  foreach s : filter_sub_caffe2_sources
    nnstreamer_filter_caffe2_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  nnstreamer_filter_caffe2_deps = [caffe2_dep, protobuf_dep, glib_dep, gst_dep, nnstreamer_dep]

  shared_library('nnstreamer_filter_caffe2',
    nnstreamer_filter_caffe2_sources,
    cpp_args: ['-Wno-sign-compare'],
    dependencies: nnstreamer_filter_caffe2_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_caffe2',
    nnstreamer_filter_caffe2_sources,
    cpp_args: ['-Wno-sign-compare'],
    dependencies: nnstreamer_filter_caffe2_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

filter_sub_python_sources = [
  'tensor_filter_python.c',
  'tensor_filter_python_core.cc'
]

nnstreamer_filter_python_sources = []
foreach s : filter_sub_python_sources
  nnstreamer_filter_python_sources += join_paths(meson.current_source_dir(), s)
endforeach

if have_python2
  nnstreamer_filter_python_deps = [python2_dep, libdl_dep, glib_dep, gst_dep, nnstreamer_dep]

  shared_library('nnstreamer_filter_python2',
    nnstreamer_filter_python_sources,
    cpp_args: ['-Wno-sign-compare'],
    dependencies: nnstreamer_filter_python_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_python2',
    nnstreamer_filter_python_sources,
    cpp_args: ['-Wno-sign-compare'],
    dependencies: nnstreamer_filter_python_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )

  shared_library('nnstreamer_python2',
    'tensor_filter_python_api.c',
    name_prefix: '',
    dependencies: nnstreamer_filter_python_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )
endif

if have_python3
  nnstreamer_filter_python_deps = [python3_dep, libdl_dep, glib_dep, gst_dep, nnstreamer_dep]

  shared_library('nnstreamer_filter_python3',
    nnstreamer_filter_python_sources,
    cpp_args: ['-Wno-sign-compare'],
    dependencies: nnstreamer_filter_python_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_python3',
    nnstreamer_filter_python_sources,
    cpp_args: ['-Wno-sign-compare'],
    dependencies: nnstreamer_filter_python_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )

  shared_library('nnstreamer_python3',
    'tensor_filter_python_api.c',
    name_prefix: '',
    dependencies: nnstreamer_filter_python_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )
endif

if get_option('enable-movidius-ncsdk2')
  # Explicitly checks mvnc.h in the ncsdk2 directory
  nns_mvncsdk2_dep = cc.find_library('mvnc', required: false)
  if not nns_mvncsdk2_dep.found()
    warning('Failed to find \'libmvnc.so\' despite setting enable-movidius-ncsdk2. This option be ignored.')
  elif not cc.check_header('mvnc2/mvnc.h')
    warning('Failed to find \'mvnc2/mvnc.h\' despite setting enable-movidius-ncsdk2. This option be ignored.')
  else
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
      c_args: ['-Wno-sign-compare'],
      dependencies: nnstreamer_filter_mvncsdk2_deps,
      install: false,
      install_dir: filter_subplugin_install_dir
    )

    static_library('nnstreamer_filter_movidius-ncsdk2',
      nnstreamer_filter_mvncsdk2_sources,
      c_args: ['-Wno-sign-compare'],
      dependencies: nnstreamer_filter_mvncsdk2_deps,
      install: false,
      install_dir: nnstreamer_libdir
    )
  endif
endif

if get_option('enable-cppfilter')
  filter_sub_cpp_sources = ['tensor_filter_cpp.cc']

  nnstreamer_filter_cpp_sources = []
  foreach s : filter_sub_cpp_sources
    nnstreamer_filter_cpp_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  shared_library('nnstreamer_filter_cpp',
    nnstreamer_filter_cpp_sources,
    dependencies: [glib_dep, gst_dep, nnstreamer_dep],
    install: true,
    install_dir: filter_subplugin_install_dir
  )
  install_headers(['tensor_filter_cpp.h'],
    subdir: 'nnstreamer'
  )
endif
