if get_option('enable-edgetpu')
  unittest_edgetpu_helper = shared_library('edgetpu',
    ['dummy_edgetpu.cc'],
    dependencies: tflite_support_deps,
    soversion: 1
  )

  unittest_edgetpu_helper_dep = declare_dependency(
    link_with: unittest_edgetpu_helper,
  )

  unittest_edgetpu_deps = [
    gtest_dep,
    glib_dep,
    gst_dep,
    gst_base_dep,
    nnstreamer_dep, # need to modify this to get the dummy loaded
    unittest_edgetpu_helper_dep,
  ]
  unittest_edgetpu = executable('unittest_edgetpu',
    ['unittest_edgetpu.cc'],
    dependencies: unittest_edgetpu_deps,
    install: get_option('install-test'),
    install_dir: unittest_install_dir
  )

  test('unittest_edgetpu', unittest_edgetpu, args: ['--gst-plugin-path=../../'])
endif
