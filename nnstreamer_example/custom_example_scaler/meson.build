nnstreamer_customfilter_scaler = shared_library('nnstreamer_customfilter_scaler',
  'nnstreamer_customfilter_example_scaler.c',
  dependencies: [nnstreamer_deps],
  include_directories: nnstreamer_inc,
  objects: [tensor_commonOBJ, tensor_filter_customOBJ]
)

nnstreamer_customfilter_scaler_allocator = shared_library('nnstreamer_customfilter_scaler_allocator',
  'nnstreamer_customfilter_example_scaler_allocator.c',
  dependencies: [nnstreamer_deps],
  include_directories: nnstreamer_inc,
  objects: [tensor_commonOBJ, tensor_filter_customOBJ]
)
