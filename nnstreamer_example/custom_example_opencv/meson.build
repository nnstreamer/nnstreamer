opencv_dep = dependency('opencv')

nnstreamer_customfilter_opencv_scaler = shared_library('nnstreamer_customfilter_opencv_scaler',
  'nnstreamer_customfilter_opencv_scaler.cc',
  dependencies: [nnstreamer_deps, opencv_dep],
  include_directories: nnstreamer_inc,
  objects: [tensor_commonOBJ, tensor_filter_customOBJ]
)

nnstreamer_customfilter_opencv_average = shared_library('nnstreamer_customfilter_opencv_average',
  'nnstreamer_customfilter_opencv_average.cc',
  dependencies: [nnstreamer_deps, opencv_dep],
  include_directories: nnstreamer_inc,
  objects: [tensor_commonOBJ, tensor_filter_customOBJ]
)
