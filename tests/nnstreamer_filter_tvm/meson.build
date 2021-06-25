unittest_filter_tvm = executable('unittest_filter_tvm',
  ['unittest_filter_tvm.cc'],
  dependencies: [unittest_util_dep, glib_dep, gst_dep, nnstreamer_dep, gtest_dep],
  install: get_option('install-test'),
  install_dir: unittest_install_dir
)

test('unittest_filter_tvm', unittest_filter_tvm, env: testenv)
