if get_option('enable-edgetpu')
  unittest_edgetpu_deps = [
    gtest_dep,
    glib_dep,
    gst_dep,
    gst_base_dep,
    nnstreamer_dep, # need to modify this to get the dummy loaded
  ]
  unittest_edgetpu = executable('unittest_edgetpu',
    ['unittest_edgetpu.cc'],
    dependencies: unittest_edgetpu_deps,
    install: get_option('install-test'),
    install_dir: unittest_install_dir
  )

  test('unittest_edgetpu', unittest_edgetpu, env: testenv)
endif
