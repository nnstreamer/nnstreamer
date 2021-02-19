test_nnfw_deps = [glib_dep, gst_dep, nnstreamer_dep, gtest_dep, nnfw_plugin_dep, unittest_util_dep]

unittest_nnfw_runtime_raw = executable('unittest_nnfw_runtime_raw',
  'unittest_tizen_nnfw_runtime_raw.cc',
  dependencies: test_nnfw_deps,
  install: get_option('install-test'),
  install_dir: unittest_install_dir,
)

test('unittest_nnfw_runtime_raw', unittest_nnfw_runtime_raw, env: testenv)
