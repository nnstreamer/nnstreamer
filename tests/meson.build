# Build nnstreamer subplugins and apps for test cases
subdir('nnstreamer_example')

# Set unittest install dir
unittest_install_dir = join_paths(unittest_base_dir, 'tests')

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
    depends: b2p,
    command: [copy, '@INPUT@', '@CURRENT_SOURCE_DIR@'],
    build_by_default: true,
    build_always_stale: true,
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
  lesser_code_quality_accepted_for_unittest_code = declare_dependency(compile_args: ['-Wno-unused-parameter', '-Wno-missing-field-initializers', '-Wno-maybe-uninitialized'])

  # Shared library of internal APIs for nnstreamer-gtest
  unittest_util_shared = shared_library('nnstreamer_unittest_util',
    join_paths(meson.current_source_dir(), 'unittest_util.c'),
    dependencies: nnstreamer_dep,
    include_directories: nnstreamer_inc,
    install: get_option('install-test'),
    install_dir: nnstreamer_libdir
  )
  unittest_util_dep = declare_dependency(link_with: unittest_util_shared,
    dependencies: nnstreamer_dep,
    compile_args: ['-DFAKEDLOG=1'],
    include_directories: include_directories('.')
  )

  nnstreamer_unittest_deps = [
    unittest_util_dep,
    glib_dep,
    gst_dep,
    gst_app_dep,
    gst_check_dep,
    gtest_dep,
    lesser_code_quality_accepted_for_unittest_code
  ]

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
      dependencies: [nnstreamer_unittest_deps],
      install: get_option('install-test'),
      install_dir: unittest_install_dir
    )

    test('unittest_sink', unittest_sink, timeout: 120, env: testenv)

    # Run unittest_plugins
    unittest_plugins = executable('unittest_plugins',
      join_paths('nnstreamer_plugins', 'unittest_plugins.cc'),
      dependencies: [nnstreamer_unittest_deps, nnstreamer_internal_deps],
      install: get_option('install-test'),
      install_dir: unittest_install_dir
    )

    test('unittest_plugins', unittest_plugins, env: testenv)

    # Run unittest_if
    unittest_if = executable('unittest_if',
      join_paths('nnstreamer_if', 'unittest_if.cc'),
      dependencies: [nnstreamer_unittest_deps],
      install: get_option('install-test'),
      install_dir: unittest_install_dir
    )

    test('unittest_if', unittest_if, env: testenv)

    # Run unittest_rate
    unittest_rate = executable('unittest_rate',
      join_paths('nnstreamer_rate', 'unittest_rate.cc'),
      dependencies: [nnstreamer_unittest_deps],
      install: get_option('install-test'),
      install_dir: unittest_install_dir
    )

    test('unittest_rate', unittest_rate, env: testenv)

    # Run unittest_latency
    unittest_latency = executable('unittest_latency',
      join_paths('nnstreamer_latency', 'unittest_latency.cc'),
      dependencies: [nnstreamer_unittest_deps],
      install: get_option('install-test'),
      install_dir: unittest_install_dir
    )

    test('unittest_latency', unittest_latency, env: testenv)

    # Run unittest_filter_single
    unittest_filter_single = executable('unittest_filter_single',
      join_paths('nnstreamer_filter_single', 'unittest_filter_single.cc'),
      dependencies: [nnstreamer_single_dep, gtest_dep, lesser_code_quality_accepted_for_unittest_code],
      install: get_option('install-test'),
      install_dir: unittest_install_dir
    )

    test('unittest_filter_single', unittest_filter_single, env: testenv)

    # Run unittest_join
    unittest_join = executable('unittest_join',
      join_paths('gstreamer_join', 'unittest_join.cc'),
      dependencies: [nnstreamer_unittest_deps, join_dep],
      install: get_option('install-test'),
      install_dir: unittest_install_dir
    )

    test('unittest_join', unittest_join, env: testenv)

    # Run unittest_mqtt
    if mqtt_support_is_available
      unittest_mqtt_w_helper = executable('unittest_mqtt_w_helper',
          join_paths('gstreamer_mqtt', 'unittest_mqtt_w_helper.cc'),
          dependencies: [gstmqtt_dep, nnstreamer_unittest_deps],
          install: get_option('install-test'),
          install_dir: unittest_install_dir)

      test('unittest_mqtt_w_helper', unittest_mqtt_w_helper, env: testenv)

      unittest_mqtt = executable('unittest_mqtt',
          join_paths('gstreamer_mqtt', 'unittest_mqtt.cc'),
          dependencies: [nnstreamer_unittest_deps],
          install: get_option('install-test'),
          install_dir: unittest_install_dir)

      test('unittest_mqtt', unittest_mqtt, env: testenv)

      # ntp util test
      gmock_dep = dependency('gmock', required: false)
      if get_option('enable-tizen') and gmock_dep.found()
        gstreamer_ntp_util_mock = executable('unittest_ntp_util',
            join_paths('gstreamer_mqtt', 'unittest_ntp_util_mock.cc'),
            dependencies: [nnstreamer_unittest_deps, gmock_dep, gstmqtt_dep],
            install: get_option('install-test'),
            install_dir: unittest_install_dir)

        test('gstreamer_ntp_util_mock', gstreamer_ntp_util_mock, env: testenv)
      endif
    endif

    # Run unittest_src_iio
    if tensor_src_iio_build
      unittest_src_iio = executable('unittest_src_iio',
        join_paths('nnstreamer_source', 'unittest_src_iio.cc'),
        dependencies: [nnstreamer_unittest_deps],
        install: get_option('install-test'),
        install_dir: unittest_install_dir
      )

      test('unittest_src_iio', unittest_src_iio, timeout: 120, env: testenv)
    endif

    # Run unittest_converter
    if flatbuf_support_is_available and have_python3
      unittest_converter = executable('unittest_converter',
        join_paths('nnstreamer_converter', 'unittest_converter.cc'),
        dependencies: [nnstreamer_unittest_deps, flatbuf_dep, nnstreamer_python3_helper_dep],
        install: get_option('install-test'),
        install_dir: unittest_install_dir
      )

      test('unittest_converter', unittest_converter, env: testenv)
    endif
  endif

  # Armnn unittest
  if armnn_support_is_available
    unittest_filter_armnn = executable('unittest_filter_armnn',
      join_paths('nnstreamer_filter_armnn', 'unittest_filter_armnn.cc'),
      dependencies: [nnstreamer_unittest_deps, armnn_plugin_dep],
      install: get_option('install-test'),
      install_dir: unittest_install_dir
    )

    test('unittest_filter_armnn', unittest_filter_armnn, env: testenv)
  endif

  # Lua unittest
  if lua_support_is_available
    unittest_filter_lua = executable('unittest_filter_lua',
      join_paths('nnstreamer_filter_lua', 'unittest_filter_lua.cc'),
      dependencies: [nnstreamer_unittest_deps],
      install: get_option('install-test'),
      install_dir: unittest_install_dir
    )

    test('unittest_filter_lua', unittest_filter_lua, env: testenv)
  endif

  # SNPE unittest
  if snpe_support_is_available
    unittest_filter_snpe = executable('unittest_filter_snpe',
      join_paths('nnstreamer_filter_snpe', 'unittest_filter_snpe.cc'),
      dependencies: [nnstreamer_unittest_deps],
      install: get_option('install-test'),
      install_dir: unittest_install_dir
    )

    test('unittest_filter_snpe', unittest_filter_snpe, env: testenv)
  endif

  # tensorflow2-lite unittest
  if tflite2_support_is_available
    unittest_filter_tensorflow2_lite = executable('unittest_filter_tensorflow2_lite',
      join_paths('nnstreamer_filter_tensorflow2_lite', 'unittest_filter_tensorflow2_lite.cc'),
      dependencies: [nnstreamer_unittest_deps],
      install: get_option('install-test'),
      install_dir: unittest_install_dir
    )

    test('unittest_filter_tensorflow2_lite', unittest_filter_tensorflow2_lite, env: testenv)
  endif

  # Run unittest_decoder
  if flatbuf_support_is_available
    unittest_decoder = executable('unittest_decoder',
      join_paths('nnstreamer_decoder', 'unittest_decoder.cc'),
      dependencies: [nnstreamer_unittest_deps, flatbuf_dep],
      install: get_option('install-test'),
      install_dir: unittest_install_dir
    )

    test('unittest_decoder', unittest_decoder, env: testenv)
  endif

  # gRPC unittest
  if grpc_support_is_available
    unittest_grpc = executable('unittest_grpc',
      join_paths('nnstreamer_grpc', 'unittest_grpc.cc'),
      dependencies: nnstreamer_unittest_deps,
      install: get_option('install-test'),
      install_dir: unittest_install_dir
    )

    test('unittest_grpc', unittest_grpc, env: testenv)
  endif

  # Tizen sensor-src
  if get_option('enable-tizen-sensor')
    subdir('tizen_sensor')
  endif

  # Tensor filter extensions basic test cases
  subdir('nnstreamer_filter_extensions_common')

  # Tizen NNFW runtime
  if nnfw_runtime_support_is_available
    subdir('tizen_nnfw_runtime')
  endif

  if (tflite_support_is_available or tflite2_support_is_available) and get_option('enable-edgetpu')
    subdir('nnstreamer_filter_edgetpu')
  endif

  if mvncsdk2_support_is_available
    subdir('nnstreamer_filter_mvncsdk2')
  endif

  if tvm_support_is_available
    subdir('nnstreamer_filter_tvm')
  endif

  if get_option('enable-cppfilter')
    subdir('cpp_methods')
  endif

  if get_option('enable-openvino')
    subdir('nnstreamer_filter_openvino')
  endif

  if mxnet_support_is_available
    subdir('nnstreamer_filter_mxnet')
  endif

  if nnstreamer_edge_support_is_available
    subdir('nnstreamer_edge')
  endif

  if tflite_support_is_available or tflite2_support_is_available
    unittest_filter_shared_model = executable('unittest_filter_shared_model',
      join_paths('nnstreamer_filter_shared_model', 'unittest_filter_shared_model.cc'),
      dependencies: [nnstreamer_unittest_deps, unittest_util_dep],
      install: get_option('install-test'),
      install_dir: unittest_install_dir
    )

    test('unittest_filter_shared_model', unittest_filter_shared_model, timeout: 30, env: testenv)
  endif
endif # gtest_dep.found()

tensor_filter_ext_enabled = tflite_support_is_available or \
    tflite2_support_is_available or \
    tf_support_is_available or \
    have_python3 or \
    pytorch_support_is_available or caffe2_support_is_available or \
    nnfw_runtime_support_is_available or get_option('enable-edgetpu') or \
    mvncsdk2_support_is_available or get_option('enable-openvino') or \
    mxnet_support_is_available
if get_option('install-test') and tensor_filter_ext_enabled
  install_subdir('test_models', install_dir: unittest_install_dir)
endif


# Install Unittest
if get_option('install-test')
  install_data('gen24bBMP.py', install_dir: unittest_install_dir)
  install_data('test_utils.py', install_dir: unittest_install_dir)
  install_subdir('nnstreamer_converter', install_dir: unittest_install_dir)
  install_subdir('nnstreamer_merge', install_dir: unittest_install_dir)
  install_subdir('nnstreamer_decoder', install_dir: unittest_install_dir)
  install_subdir('nnstreamer_decoder_boundingbox', install_dir: unittest_install_dir)
  install_subdir('nnstreamer_decoder_pose', install_dir: unittest_install_dir)
  install_subdir('nnstreamer_demux', install_dir: unittest_install_dir)
  install_subdir('nnstreamer_filter_custom', install_dir: unittest_install_dir)
  if tflite_support_is_available
    install_subdir('nnstreamer_filter_tensorflow_lite', install_dir: unittest_install_dir)
    install_subdir('nnstreamer_decoder_image_labeling', install_dir: unittest_install_dir)
    install_subdir('nnstreamer_decoder_image_segment', install_dir: unittest_install_dir)
    install_subdir('nnstreamer_filter_reload', install_dir: unittest_install_dir)
  endif
  if tflite2_support_is_available
    install_subdir('nnstreamer_filter_tensorflow2_lite', install_dir: unittest_install_dir)
  endif
  if have_python3
    if flatbuf_support_is_available
      install_subdir('nnstreamer_converter_python3', install_dir: unittest_install_dir)
      install_subdir('nnstreamer_decoder_python3', install_dir: unittest_install_dir)
    endif
    install_subdir('nnstreamer_filter_python3', install_dir: unittest_install_dir)
  endif
  if tf_support_is_available
    install_subdir('nnstreamer_filter_tensorflow', install_dir: unittest_install_dir)
  endif
  if pytorch_support_is_available
    install_subdir('nnstreamer_filter_pytorch', install_dir: unittest_install_dir)
  endif
  if caffe2_support_is_available
    install_subdir('nnstreamer_filter_caffe2', install_dir: unittest_install_dir)
  endif
  if deepview_rt_support_is_available
    install_subdir('nnstreamer_filter_deepview_rt', install_dir: unittest_install_dir)
  endif
  if flatbuf_support_is_available
    install_subdir('nnstreamer_flatbuf', install_dir: unittest_install_dir)
    install_subdir('nnstreamer_flexbuf', install_dir: unittest_install_dir)
  endif
  if protobuf_support_is_available
    install_subdir('nnstreamer_protobuf', install_dir: unittest_install_dir)
  endif
  if mxnet_support_is_available
    install_subdir('unittest_filter_mxnet', install_dir: unittest_install_dir)
  endif
  install_subdir('nnstreamer_mux', install_dir: unittest_install_dir)
  install_subdir('nnstreamer_rate', install_dir: unittest_install_dir)
  install_subdir('nnstreamer_repo', install_dir: unittest_install_dir)
  install_subdir('nnstreamer_repo_dynamicity', install_dir: unittest_install_dir)
  install_subdir('nnstreamer_repo_lstm', install_dir: unittest_install_dir)
  install_subdir('nnstreamer_repo_rnn', install_dir: unittest_install_dir)
  install_subdir('nnstreamer_split', install_dir: unittest_install_dir)
  install_subdir('transform_arithmetic', install_dir: unittest_install_dir)
  install_subdir('transform_clamp', install_dir: unittest_install_dir)
  install_subdir('transform_dimchg', install_dir: unittest_install_dir)
  install_subdir('transform_stand', install_dir: unittest_install_dir)
  install_subdir('transform_transpose', install_dir: unittest_install_dir)
  install_subdir('transform_typecast', install_dir: unittest_install_dir)
endif
