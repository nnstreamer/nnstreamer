tensor_split_sources = [
  'gsttensorsplit.c'
]

tensor_splitOBJ = static_library('tensor_split',
  tensor_split_sources,
  c_args: nnstreamer_base_args,
  dependencies: [nnstreamer_deps],
  include_directories: nnstreamer_inc
).extract_all_objects()

objects += tensor_splitOBJ
