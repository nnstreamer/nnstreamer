tensor_aggregator_sources = [
  'tensor_aggregator.c'
]

tensor_aggregatorOBJ = static_library('tensor_aggregator',
  tensor_aggregator_sources,
  c_args: nnstreamer_base_args,
  dependencies: [nnstreamer_deps],
  include_directories: nnstreamer_inc
).extract_all_objects()

objects += tensor_aggregatorOBJ
