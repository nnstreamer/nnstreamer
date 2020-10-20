# Build and copy exe for ssat
copy = find_program('cp', required: true)
libpng_dep = dependency('libpng', required: false)
if libpng_dep.found()
  b2p = executable('bmp2png',
    'bmp2png.c',
    dependencies: [libpng_dep, glib_dep],
    install: get_option('install-test'),
    install_dir: unittest_install_dir
  )

  custom_target('copy-bmp2png',
    input: b2p,
    output: 'b2p',
    command: [copy, '@INPUT@', meson.current_source_dir()],
    build_by_default: true
  )
endif

# ssat repo_dynamic
subdir('nnstreamer_repo_dynamicity')

# filter_reload (Currently, the reload test for tensor filter requires tflite)
if tflite_support_is_available
  subdir('nnstreamer_filter_reload')
endif

# gtest
gtest_dep = dependency('gtest', required: false)
if gtest_dep.found()
  nnstreamer_unittest_deps = [
    nnstreamer_dep,
    nnstreamer_internal_deps,
    glib_dep,
    gst_dep,
    gst_app_dep,
    gst_check_dep,
    gtest_dep
  ]

  # Shared library of internal APIs for nnstreamer-gtest
  unittest_util_shared = shared_library('nnstreamer_unittest_util',
    join_paths(meson.current_source_dir(), 'unittest_util.c'),
    dependencies: nnstreamer_base_deps,
    include_directories: nnstreamer_inc,
    install: get_option('install-test'),
    install_dir: nnstreamer_libdir
  )
  unittest_util_dep = declare_dependency(link_with: unittest_util_shared,
    dependencies: nnstreamer_base_deps,
    include_directories: include_directories('.')
  )

  # Run unittest_common
  unittest_common = executable('unittest_common',
    join_paths('common', 'unittest_common.cc'),
    dependencies: [nnstreamer_unittest_deps],
    install: get_option('install-test'),
    install_dir: unittest_install_dir
  )

  test('unittest_common', unittest_common, env: testenv)

  # Run unittest_sink
  gst18_dep = dependency('gstreamer-' + gst_api_verision, version : '>=1.8', required : false)
  if gst18_dep.found()
    unittest_sink = executable('unittest_sink',
      join_paths('nnstreamer_sink', 'unittest_sink.cc'),
      dependencies: [nnstreamer_unittest_deps, unittest_util_dep],
      install: get_option('install-test'),
      install_dir: unittest_install_dir
    )

    test('unittest_sink', unittest_sink, timeout: 120, env: testenv)

    # Run unittest_plugins
    unittest_plugins = executable('unittest_plugins',
      join_paths('nnstreamer_plugins', 'unittest_plugins.cc'),
      dependencies: [nnstreamer_unittest_deps],
      install: get_option('install-test'),
      install_dir: unittest_install_dir
    )

    test('unittest_plugins', unittest_plugins, env: testenv)

  # Run unittest_if
    unittest_if = executable('unittest_if',
      join_paths('nnstreamer_if', 'unittest_if.cc'),
      dependencies: [nnstreamer_unittest_deps, unittest_util_dep],
      install: get_option('install-test'),
      install_dir: unittest_install_dir
    )

    test('unittest_if', unittest_if, env: testenv)

  # Run unittest_src_iio
    if build_platform != 'macos'
      unittest_src_iio = executable('unittest_src_iio',
        join_paths('nnstreamer_source', 'unittest_src_iio.cc'),
        dependencies: [nnstreamer_unittest_deps, unittest_util_dep],
        install: get_option('install-test'),
        install_dir: unittest_install_dir
      )

      test('unittest_src_iio', unittest_src_iio, timeout: 120, env: testenv)
    endif
  endif

  # Armnn unittest
  if armnn_support_is_available
    unittest_filter_armnn = executable('unittest_filter_armnn',
      join_paths('nnstreamer_filter_armnn', 'unittest_filter_armnn.cc'),
      dependencies: [glib_dep, gst_dep, nnstreamer_dep, gtest_dep, armnn_plugin_dep],
      install: get_option('install-test'),
      install_dir: unittest_install_dir
    )

    test('unittest_filter_armnn', unittest_filter_armnn, env: testenv)
  endif

  # Tizen C-API
  if get_option('enable-capi')
    subdir('tizen_capi')
  endif

  # Tensor filter extensions basic test cases
  subdir('nnstreamer_filter_extensions_common')

  # Tizen NNFW runtime
  if nnfw_runtime_support_is_available
    subdir('tizen_nnfw_runtime')
  endif

  if tflite_support_is_available and get_option('enable-edgetpu')
    subdir('nnstreamer_filter_edgetpu')
  endif

  if mvncsdk2_support_is_available
    subdir('nnstreamer_filter_mvncsdk2')
  endif

  if get_option('enable-cppfilter')
    subdir('cpp_methods')
  endif

  if get_option('enable-openvino')
    subdir('nnstreamer_filter_openvino')
  endif
endif # gtest_dep.found()

# Install data required for unittest
unittest_tests_install_dir = join_paths(unittest_install_dir,'tests')

tensor_filter_ext_enabled = tflite_support_is_available or \
    tf_support_is_available or \
    have_python2 or have_python3 or \
    pytorch_support_is_available or caffe2_support_is_available or \
    nnfw_runtime_support_is_available or get_option('enable-edgetpu') or \
    mvncsdk2_support_is_available or get_option('enable-openvino')
if get_option('install-test') and (get_option('enable-capi') or tensor_filter_ext_enabled)
  install_subdir('test_models', install_dir: unittest_tests_install_dir)
endif


# Install Unittest
if get_option('install-test')
  install_data('gen24bBMP.py', install_dir: unittest_tests_install_dir)
  install_subdir('nnstreamer_converter', install_dir: unittest_tests_install_dir)
  install_subdir('nnstreamer_merge', install_dir: unittest_tests_install_dir)
  install_subdir('nnstreamer_decoder', install_dir: unittest_tests_install_dir)
  install_subdir('nnstreamer_demux', install_dir: unittest_tests_install_dir)
  install_subdir('nnstreamer_filter_custom', install_dir: unittest_tests_install_dir)
  if tflite_support_is_available
    install_subdir('nnstreamer_filter_tensorflow_lite', install_dir: unittest_tests_install_dir)
    install_subdir('nnstreamer_decoder_image_labeling', install_dir: unittest_tests_install_dir)
    install_subdir('nnstreamer_filter_reload', install_dir: unittest_tests_install_dir)
  endif
  if tflite2_support_is_available
    install_subdir('nnstreamer_filter_tensorflow2_lite', install_dir: unittest_tests_install_dir)
  endif
  if have_python2 or have_python3
    install_subdir('nnstreamer_filter_python', install_dir: unittest_tests_install_dir)
  endif
  if tf_support_is_available
    install_subdir('nnstreamer_filter_tensorflow', install_dir: unittest_tests_install_dir)
  endif
  if pytorch_support_is_available
    install_subdir('nnstreamer_filter_pytorch', install_dir: unittest_tests_install_dir)
  endif
  if caffe2_support_is_available
    install_subdir('nnstreamer_filter_caffe2', install_dir: unittest_tests_install_dir)
  endif
  if mvncsdk2_support_is_available
    install_subdir('unittest_filter_mvncsdk2', install_dir: unittest_tests_install_dir)
  endif
  install_subdir('nnstreamer_mux', install_dir: unittest_tests_install_dir)
  install_subdir('nnstreamer_rate', install_dir: unittest_tests_install_dir)
  install_subdir('nnstreamer_repo', install_dir: unittest_tests_install_dir)
  install_subdir('nnstreamer_repo_dynamicity', install_dir: unittest_tests_install_dir)
  install_subdir('nnstreamer_repo_lstm', install_dir: unittest_tests_install_dir)
  install_subdir('nnstreamer_repo_rnn', install_dir: unittest_tests_install_dir)
  install_subdir('nnstreamer_split', install_dir: unittest_tests_install_dir)
  install_subdir('transform_arithmetic', install_dir: unittest_tests_install_dir)
  install_subdir('transform_dimchg', install_dir: unittest_tests_install_dir)
  install_subdir('transform_stand', install_dir: unittest_tests_install_dir)
  install_subdir('transform_transpose', install_dir: unittest_tests_install_dir)
  install_subdir('transform_typecast', install_dir: unittest_tests_install_dir)
  if get_option('enable-cppfilter')
    install_subdir('cpp_methods', install_dir: unittest_tests_install_dir)
  endif
endif
