if grpc_support_is_available
  grpc_tensor_sink_source_files = [
    'tensor_sink_grpc.c'
  ]
  grpc_tensor_sink_sources = []

  foreach s : grpc_tensor_sink_source_files
    grpc_tensor_sink_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  tensor_sink_grpc_dep = declare_dependency(
    sources : grpc_tensor_sink_sources,
    dependencies : [glib_dep, gst_dep, nnstreamer_dep, grpc_support_deps, protobuf_util_dep]
  )
endif
