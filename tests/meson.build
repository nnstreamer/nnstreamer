# These custom filters are used in unittest_sink.
library('nnscustom_framecounter',
  join_paths('nnstreamer_sink', 'nnscustom_framecounter.c'),
  dependencies: [glib_dep, gst_dep, nnstreamer_dep],
  install: get_option('install-test'),
  install_dir: customfilter_install_dir
)

library('nnscustom_drop_buffer',
  join_paths('nnstreamer_sink', 'nnscustom_drop_buffer.c'),
  dependencies: [glib_dep, gst_dep, nnstreamer_dep],
  install: get_option('install-test'),
  install_dir: customfilter_install_dir
)

# Build and copy exe for ssat
libpng_dep = dependency('libpng', required: false)
if libpng_dep.found()
  b2p = executable('bmp2png',
    'bmp2png.c',
    dependencies: [libpng_dep, glib_dep],
    install: get_option('install-test'),
    install_dir: unittest_install_dir
  )

  copy = find_program('cp')
  custom_target('copy-bmp2png',
    input: b2p,
    output: 'b2p',
    command: [copy, '@INPUT@', meson.current_source_dir()],
    build_by_default: true
  )
endif

# ssat repo_dynamic
executable('unittest_repo',
  join_paths('nnstreamer_repo_dynamicity', 'tensor_repo_dynamic_test.c'),
  dependencies: [nnstreamer_dep, gst_dep, glib_dep],
  install: get_option('install-test'),
  install_dir: unittest_install_dir
)

# filter_reload
executable('unittest_filter_reload',
  join_paths('nnstreamer_filter_reload', 'tensor_filter_reload_test.c'),
  dependencies: [nnstreamer_dep, gst_dep, glib_dep],
  install: get_option('install-test'),
  install_dir: unittest_install_dir
)

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

  # Run unittest_common
  unittest_common = executable('unittest_common',
    join_paths('common', 'unittest_common.cc'),
    dependencies: [nnstreamer_unittest_deps],
    install: get_option('install-test'),
    install_dir: unittest_install_dir
  )

  test('unittest_common', unittest_common)

  # Run unittest_sink
  unittest_sink = executable('unittest_sink',
    join_paths('nnstreamer_sink', 'unittest_sink.cc'),
    dependencies: [nnstreamer_unittest_deps],
    install: get_option('install-test'),
    install_dir: unittest_install_dir
  )

  test('unittest_sink', unittest_sink, timeout: 120, args: ['--gst-plugin-path=..'])

  # Run unittest_plugins
  unittest_plugins = executable('unittest_plugins',
    join_paths('nnstreamer_plugins', 'unittest_plugins.cc'),
    dependencies: [nnstreamer_unittest_deps],
    install: get_option('install-test'),
    install_dir: unittest_install_dir
  )

  test('unittest_plugins', unittest_plugins, args: ['--gst-plugin-path=..'])

  # Run unittest_src_iio
  if build_platform != 'macos'
    unittest_src_iio = executable('unittest_src_iio',
      join_paths('nnstreamer_source', 'unittest_src_iio.cc'),
      dependencies: [nnstreamer_unittest_deps],
      install: get_option('install-test'),
      install_dir: unittest_install_dir
    )

    test('unittest_src_iio', unittest_src_iio, timeout: 120, args: ['--gst-plugin-path=..'])
  endif

  # Armnn unittest
  if get_option('enable-armnn')
    unittest_filter_armnn = executable('unittest_filter_armnn',
      join_paths('nnstreamer_filter_armnn', 'unittest_filter_armnn.cc'),
      dependencies: [glib_dep, gst_dep, nnstreamer_dep, gtest_dep, armnn_plugin_dep],
      install: get_option('install-test'),
      install_dir: unittest_install_dir
    )

    test('unittest_filter_armnn', unittest_filter_armnn, args: ['--gst-plugin-path=../..'])
  endif

endif

# Install data required for unittest
unittest_tests_install_dir = join_paths(unittest_install_dir,'tests')

tensor_filter_ext_enabled = get_option('enable-tensorflow-lite') or \
    get_option('enable-python') or get_option('enable-tensorflow') or \
    get_option('enable-pytorch') or get_option('enable-caffe2') or \
    get_option('enable-nnfw-runtime') or get_option('enable-edgetpu')
if get_option('install-test') and (get_option('enable-capi') or tensor_filter_ext_enabled)
  install_subdir('test_models', install_dir: unittest_tests_install_dir)
endif

# Tizen C-API
if get_option('enable-capi')
  subdir('tizen_capi')
endif

# Tizen NNFW runtime
if get_option('enable-nnfw-runtime')
  subdir('tizen_nnfw_runtime')
endif

if get_option('enable-tensorflow-lite') and get_option('enable-edgetpu')
  subdir('nnstreamer_filter_edgetpu')
endif

if get_option('enable-cppfilter')
  subdir('cpp_methods')
endif

# Tensor filter extensions basic test cases
subdir('nnstreamer_filter_extensions_common')

# Install Unittest
if get_option('install-test')
  install_data('gen24bBMP.py', install_dir: unittest_tests_install_dir)
  install_subdir('nnstreamer_converter', install_dir: unittest_tests_install_dir)
  install_subdir('nnstreamer_merge', install_dir: unittest_tests_install_dir)
  install_subdir('nnstreamer_decoder', install_dir: unittest_tests_install_dir)
  install_subdir('nnstreamer_demux', install_dir: unittest_tests_install_dir)
  install_subdir('nnstreamer_filter_custom', install_dir: unittest_tests_install_dir)
  if get_option('enable-tensorflow-lite')
    install_subdir('nnstreamer_filter_tensorflow_lite', install_dir: unittest_tests_install_dir)
    install_subdir('nnstreamer_decoder_image_labeling', install_dir: unittest_tests_install_dir)
    install_subdir('nnstreamer_filter_reload', install_dir: unittest_tests_install_dir)
  endif
  if get_option('enable-python')
    install_subdir('nnstreamer_filter_python', install_dir: unittest_tests_install_dir)
  endif
  if get_option('enable-tensorflow')
    install_subdir('nnstreamer_filter_tensorflow', install_dir: unittest_tests_install_dir)
  endif
  if get_option('enable-pytorch')
    install_subdir('nnstreamer_filter_pytorch', install_dir: unittest_tests_install_dir)
  endif
  if get_option('enable-caffe2')
    install_subdir('nnstreamer_filter_caffe2', install_dir: unittest_tests_install_dir)
  endif
  install_subdir('nnstreamer_mux', install_dir: unittest_tests_install_dir)
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
