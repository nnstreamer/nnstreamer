tensor_mux_sources = [
  'gsttensormux.c'
]

tensor_muxOBJ = static_library('tensor_mux',
  tensor_mux_sources,
  c_args: nnstreamer_base_args,
  dependencies: [nnstreamer_deps],
  include_directories: nnstreamer_inc
).extract_all_objects()

objects += tensor_muxOBJ
