if mxnet_support_is_available
  test_mxnet_deps = [
    nnstreamer_dep,
    nnstreamer_filter_mxnet_dep,
    gtest_dep,
    mxnet_support_deps,
    cc.find_library('opencv_core', required: true)
  ]

  test_filter_mxnet = executable ('simple_test_mxnet',
    ['simple_test_mxnet.cc'],
    dependencies : test_mxnet_deps,
    cpp_args: ['-Wno-non-virtual-dtor'],
  )

  configure_file(input : 'download_and_run.sh', output : 'download_and_run.sh',
    configuration : configuration_data()
  )
endif
