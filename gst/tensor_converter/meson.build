tensor_converter_sources = [
  'tensor_converter.c'
]

tensor_converterOBJ = static_library('tensor_converter',
  tensor_converter_sources,
  c_args: nnstreamer_base_args,
  dependencies: [nnstreamer_deps],
  include_directories: nnstreamer_inc
).extract_all_objects()

objects += tensor_converterOBJ
