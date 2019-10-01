unittest_nnfw_runtime_raw = executable('unittest_nnfw_runtime_raw',
  'unittest_tizen_nnfw_runtime_raw.cpp',
  dependencies: [glib_dep, gst_dep, nnstreamer_dep, gtest_dep, nnfw_plugin_dep],
  install: get_option('install-test'),
  install_dir: unittest_install_dir,
)

test('unittest_nnfw_runtime_raw', unittest_nnfw_runtime_raw, args: ['--gst-plugin-path=../..'])
