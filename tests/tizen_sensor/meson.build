unittest_sensor_helper = shared_library('capi-system-sensor',
  ['dummy_sensor.c'],
  dependencies: [glib_dep],
)

unittest_sensor_helper_dep = declare_dependency(
  link_with: unittest_sensor_helper,
)

unittest_tizen_sensor = executable('unittest_tizen_sensor',
  ['unittest_tizen_sensor.cc'],
  dependencies: [nnstreamer_unittest_deps, unittest_sensor_helper_dep],
  install: get_option('install-test'),
  install_dir: unittest_install_dir
)

test('unittest_tizen_sensor', unittest_tizen_sensor, env: testenv)
