if grpc_support_is_available
  grpc_tensor_sink_source_files = [
    'tensor_sink_grpc.c'
  ]

  tensor_sink_grpc_dep = declare_dependency(
    sources : grpc_tensor_sink_source_files,
    dependencies : grpc_util_dep,
  )
endif
