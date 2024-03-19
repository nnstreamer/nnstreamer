# Run unittest_datareposink
unittest_datareposink = executable('unittest_datareposink',
  'unittest_datareposink.cc',
  dependencies: [nnstreamer_unittest_deps],
  install: get_option('install-test'),
  install_dir: unittest_install_dir
)
test('unittest_datareposink', unittest_datareposink, env: testenv)

# Run unittest_datareposrc
unittest_datareposrc = executable('unittest_datareposrc',
  'unittest_datareposrc.cc',
  dependencies: [nnstreamer_unittest_deps],
  install: get_option('install-test'),
  install_dir: unittest_install_dir
)
test('unittest_datareposrc', unittest_datareposrc, env: testenv)
