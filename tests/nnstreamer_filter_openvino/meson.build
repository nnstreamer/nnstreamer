if get_option('enable-openvino')
  unittest_openvino_deps = [
    nnstreamer_unittest_deps,
    nnstreamer_filter_openvino_dep,
    openvino_deps,
  ]

  unittest_openvino_deps += declare_dependency(
    compile_args : openvino_cpp_args,
  )

  unittest_filter_openvino_helper = executable ('unittest_openvino',
    ['unittest_openvino.cc'],
    dependencies : [unittest_openvino_deps],
  )
endif
