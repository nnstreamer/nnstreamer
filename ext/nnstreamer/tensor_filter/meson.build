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
  # check_header() requires meson >= 0.47
  # if not cc.check_header('mvnc2/mvnc.h')
  nns_mvncsdk2_dep = cc.find_library('mvnc')
  if not cc.has_header('mvnc2/mvnc.h')
    # warning() requires meson >= 0.44
    # warning('enable-movidius-ncsdk2 is set but could not find \'mvnc2/mvnc.h\'.')
    message('enable-movidius-ncsdk2 is set but could not find \'mvnc2/mvnc.h\'.')
  elif not nns_mvncsdk2_dep.found()
    # warning() requires meson >= 0.44
    # warning('enable-movidius-ncsdk2 is set but could not find \'libmvnc.so\'.')
    message('enable-movidius-ncsdk2 is set but could not find \'libmvnc.so\'.')
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
