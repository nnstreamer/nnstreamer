tensor_filter_args = [nnstreamer_base_args]
tensor_filterOBJ = []

if get_option('ENABLE_TENSORFLOW_LITE')
  nnstreamer_deps += [dependency('tensorflow-lite')]
  tensor_filter_args += ['-DENABLE_TENSORFLOW_LITE']

  tensor_filter_tflitecoreOBJ = static_library('tensor_filter_tflitecore',
    ['tensor_filter_tensorflow_lite.c', 'tensor_filter_tensorflow_lite_core.cc'],
    c_args: tensor_filter_args,
    cpp_args: tensor_filter_args + ['-Wno-sign-compare'],
    dependencies: [nnstreamer_deps],
    include_directories: nnstreamer_inc
  ).extract_all_objects()

  tensor_filterOBJ += tensor_filter_tflitecoreOBJ
endif

if get_option('ENABLE_TENSORFLOW')
  nnstreamer_deps += [dependency('tensorflow'), dependency('protobuf')]
  tensor_filter_args += ['-DENABLE_TENSORFLOW']

  tensor_filter_tfcoreOBJ = static_library('tensor_filter_tfcore',
    ['tensor_filter_tensorflow.c', 'tensor_filter_tensorflow_core.cc'],
    c_args: tensor_filter_args,
    cpp_args: tensor_filter_args + ['-Wno-sign-compare'],
    dependencies: [nnstreamer_deps],
    include_directories: nnstreamer_inc
  ).extract_all_objects()

  tensor_filterOBJ += tensor_filter_tfcoreOBJ
endif

tensor_filter_mainOBJ = static_library('tensor_filter',
  ['tensor_filter.c', 'tensor_filter_custom.c'],
  c_args: tensor_filter_args,
  dependencies: [nnstreamer_deps],
  include_directories: nnstreamer_inc
).extract_all_objects()

tensor_filterOBJ += tensor_filter_mainOBJ

objects += tensor_filterOBJ
