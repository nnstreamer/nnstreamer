tensor_reposink_sources = [
  'tensor_reposink.c'
]

tensor_reposinkOBJ = static_library('tensor_reposink',
  tensor_reposink_sources,
  c_args: nnstreamer_base_args,
  dependencies: [nnstreamer_deps],
  include_directories: nnstreamer_inc
).extract_all_objects()

objects += tensor_reposinkOBJ
