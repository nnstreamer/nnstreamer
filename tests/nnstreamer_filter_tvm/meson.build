unittest_filter_tvm = executable('unittest_filter_tvm',
  ['unittest_filter_tvm.cc'],
  dependencies: [nnstreamer_unittest_deps],
  install: get_option('install-test'),
  install_dir: unittest_install_dir
)

test('unittest_filter_tvm', unittest_filter_tvm, env: testenv)
