tensor_sink_sources = [
  'tensor_sink.c'
]

tensor_sinkOBJ = static_library('tensor_sink',
  tensor_sink_sources,
  c_args: nnstreamer_base_args,
  dependencies: [nnstreamer_deps],
  include_directories: nnstreamer_inc
).extract_all_objects()

objects += tensor_sinkOBJ
