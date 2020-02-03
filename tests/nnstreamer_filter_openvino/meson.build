if get_option('enable-openvino')
  unittest_openvino_deps = [
    nnstreamer_dep,
    nnstreamer_filter_openvino_dep,
    gtest_dep
  ]

  unittest_openvino_deps += declare_dependency(
    compile_args : openvino_cpp_args,
  )

  unittest_filter_openvino_helper = executable ('unittest_openvino',
    ['unittest_openvino.cc'],
    dependencies : [unittest_openvino_deps],
  )
endif
