if get_option('enable-cppfilter') and get_option('enable-capi')
  test_deps = [
    nnstreamer_capi_dep,
    gtest_dep,
    glib_dep,
    nnstreamer_cpp_dep,
  ]

  lib_ucf = declare_dependency(compile_args: '-DTHIS_IS_SHLIB')

  lib_unittest_cppfilter = shared_library('cppfilter_test',
    join_paths(meson.current_source_dir(), 'unittest_cpp_methods.cc'),
    dependencies: [test_deps, lib_ucf, unittest_util_dep]
  )

  unittest_cppfilter = executable('unittest_cppfilter',
    join_paths(meson.current_source_dir(), 'unittest_cpp_methods.cc'),
    dependencies: [test_deps, unittest_util_dep],
    install: get_option('install-test'),
    install_dir: unittest_install_dir
  )

  test('unittest_cppfilter', unittest_cppfilter, args: ['--gst-plugin-path=../..'])
endif
