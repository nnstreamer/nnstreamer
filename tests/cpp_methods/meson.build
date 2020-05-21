cppfilter_test_lib = library('cppfilter_test',
  join_paths(meson.current_source_dir(), 'cppfilter_test.cc'),
  dependencies: [nnstreamer_cpp_dep],
  install: get_option('install-test'),
  install_dir: unittest_install_dir
)

unittest_cppfilter = executable('unittest_cppfilter',
  join_paths(meson.current_source_dir(), 'unittest_cpp_methods.cc'),
  dependencies: [gtest_dep, unittest_util_dep, nnstreamer_cpp_dep],
  link_with: cppfilter_test_lib,
  install: get_option('install-test'),
  install_dir: unittest_install_dir
)

test('unittest_cppfilter', unittest_cppfilter, env: testenv)
