nnstreamer_customfilter_passthrough = shared_library('nnstreamer_customfilter_passthrough',
  'nnstreamer_customfilter_example_passthrough.c',
  dependencies: [nnstreamer_deps],
  include_directories: nnstreamer_inc,
  objects: [tensor_commonOBJ, tensor_filter_customOBJ]
)

nnstreamer_customfilter_passthrough_variable = shared_library('nnstreamer_customfilter_passthrough_variable',
  'nnstreamer_customfilter_example_passthrough_variable.c',
  dependencies: [nnstreamer_deps],
  include_directories: nnstreamer_inc,
  objects: [tensor_commonOBJ, tensor_filter_customOBJ]
)
