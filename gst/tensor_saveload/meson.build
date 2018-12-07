tensor_saveload_sources = [
  'tensor_load.c'
]

tensor_saveloadOBJ = static_library('tensor_saveload',
  tensor_saveload_sources,
  c_args: nnstreamer_base_args,
  dependencies: [nnstreamer_deps],
  include_directories: nnstreamer_inc
).extract_all_objects()

objects += tensor_saveloadOBJ
