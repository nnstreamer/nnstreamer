tizen_apptest_deps = [
  nnstreamer_capi_dep,
  gtest_dep,
  glib_dep
]

unittest_tizen_capi = executable('unittest_tizen_capi',
  'unittest_tizen_capi.cc',
  dependencies: [tizen_apptest_deps, unittest_util_dep],
  install: get_option('install-test'),
  install_dir: unittest_install_dir
)
test('unittest_tizen_capi', unittest_tizen_capi, env: testenv)

unittest_tizen_capi_latency = executable('unittest_tizen_capi_latency',
  'unittest_tizen_capi_latency.cc',
  dependencies: [tizen_apptest_deps, unittest_util_dep],
  install: false,
  install_dir: unittest_install_dir
)
test('unittest_tizen_capi_latency', unittest_tizen_capi_latency, env: testenv)

if get_option('enable-tizen-sensor')

  unittest_sensor_helper = shared_library('capi-system-sensor',
    ['dummy_sensor.c'],
    dependencies: [glib_dep],
  )
  unittest_sensor_helper_dep = declare_dependency(
    link_with: unittest_sensor_helper,
  )

  tizen_sensor_apptest_deps = [
    gtest_dep,
    glib_dep,
    gst_dep,
    gst_base_dep,
    nnstreamer_dep,
    nnstreamer_capi_dep,
    unittest_sensor_helper_dep,
  ]
  unittest_tizen_sensor = executable('unittest_tizen_sensor',
    ['unittest_tizen_sensor.cc'],
    dependencies: [tizen_sensor_apptest_deps, unittest_util_dep],
    install: get_option('install-test'),
    install_dir: unittest_install_dir
  )
  tizen_sensor_testenv = environment()
  tizen_sensor_testenv.set('GST_PLUGIN_PATH', \
      join_paths(meson.build_root(), 'ext', 'nnstreamer', 'tensor_source'), \
      path_gst_plugin)
  tizen_sensor_testenv.set('NNSTREAMER_CONF', path_nns_conf)
  tizen_sensor_testenv.set('NNSTREAMER_FILTERS', path_nns_plugin_filters)
  tizen_sensor_testenv.set('NNSTREAMER_DECODERS', path_nns_plugin_decoders)
  tizen_sensor_testenv.set('NNSTREAMER_CONVERTERS', path_nns_plugin_converters)
  tizen_sensor_testenv.set('NNSTREAMER_SOURCE_ROOT_PATH', meson.source_root())
  test('unittest_tizen_sensor', unittest_tizen_sensor, env: tizen_sensor_testenv)
endif
