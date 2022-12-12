cppfilter_test_lib = library('cppfilter_test',
  'cppfilter_test.cc',
  dependencies: [nnstreamer_cpp_dep],
  install: get_option('install-test'),
  install_dir: nnstreamer_libdir
)

unittest_cppfilter = executable('unittest_cppfilter',
  'unittest_cpp_methods.cc',
  dependencies: [nnstreamer_unittest_deps, nnstreamer_cpp_dep],
  link_with: cppfilter_test_lib,
  install: get_option('install-test'),
  install_dir: unittest_install_dir
)

test('unittest_cppfilter', unittest_cppfilter, env: testenv)
