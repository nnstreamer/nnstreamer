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
  nnstreamer_dep,
  unittest_sensor_helper_dep,
  unittest_util_dep
]

unittest_tizen_sensor = executable('unittest_tizen_sensor',
  ['unittest_tizen_sensor.cc'],
  dependencies: tizen_sensor_apptest_deps,
  install: get_option('install-test'),
  install_dir: unittest_install_dir
)

test('unittest_tizen_sensor', unittest_tizen_sensor, env: testenv)
