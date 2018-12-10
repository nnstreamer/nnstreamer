tensor_filter_args = [nnstreamer_base_args]
tensor_filterOBJ = []

# For custom filter
tensor_filter_customOBJ = static_library('tensor_filter_custom',
  'tensor_filter_custom.c',
  c_args: tensor_filter_args,
  dependencies: [nnstreamer_deps],
  include_directories: nnstreamer_inc
).extract_all_objects()

tensor_filterOBJ += tensor_filter_customOBJ

if get_option('ENABLE_TENSORFLOW_LITE')
  tflite_dep = dependency('tensorflow-lite', required: true)

  nnstreamer_deps += [tflite_dep]
  tensor_filter_args += ['-DENABLE_TENSORFLOW_LITE']

  tensor_filter_tensorflow_liteOBJ = static_library('tensor_filter_tensorflow_lite',
    ['tensor_filter_tensorflow_lite.c', 'tensor_filter_tensorflow_lite_core.cc'],
    c_args: tensor_filter_args,
    cpp_args: tensor_filter_args + ['-Wno-sign-compare'],
    dependencies: [nnstreamer_deps],
    include_directories: nnstreamer_inc
  ).extract_all_objects()

  tensor_filterOBJ += tensor_filter_tensorflow_liteOBJ
endif

if get_option('ENABLE_TENSORFLOW')
  tf_dep = dependency('tensorflow', required: true)
  protobuf_dep = dependency('protobuf', version: '>= 3.6.0', required: true)

  nnstreamer_deps += [tf_dep, protobuf_dep]
  tensor_filter_args += ['-DENABLE_TENSORFLOW']

  tensor_filter_tensorflowOBJ = static_library('tensor_filter_tensorflow',
    ['tensor_filter_tensorflow.c', 'tensor_filter_tensorflow_core.cc'],
    c_args: tensor_filter_args,
    cpp_args: tensor_filter_args + ['-Wno-sign-compare'],
    dependencies: [nnstreamer_deps],
    include_directories: nnstreamer_inc
  ).extract_all_objects()

  tensor_filterOBJ += tensor_filter_tensorflowOBJ
endif

tensor_filter_mainOBJ = static_library('tensor_filter_main',
  'tensor_filter.c',
  c_args: tensor_filter_args,
  dependencies: [nnstreamer_deps],
  include_directories: nnstreamer_inc
).extract_all_objects()

tensor_filterOBJ += tensor_filter_mainOBJ

objects += tensor_filterOBJ
