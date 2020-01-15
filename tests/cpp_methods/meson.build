if get_option('enable-cppfilter') and get_option('enable-capi')
  test_deps = [
    nnstreamer_capi_dep,
    gtest_dep,
    glib_dep,
    nnstreamer_cpp_dep,
  ]

  unittest_cppfilter = executable('unittest_cppfilter',
    'unittest_cpp_methods.cc',
    dependencies: [test_deps],
    install: get_option('install-test'),
    install_dir: unittest_install_dir
  )

  test('unittest_cppfilter', unittest_cppfilter, args: ['--gst-plugin-path=../..'])
endif
