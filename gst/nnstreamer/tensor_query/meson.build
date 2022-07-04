tensor_query_sources = [
  'tensor_query_common.c',
  'tensor_query_serversrc.c',
  'tensor_query_serversink.c',
  'tensor_query_client.c',
  'tensor_query_server.c',
]

# tensor_query has dependency to nnstreamer-edge.
if nnstreamer_edge_support_is_available
  foreach s : tensor_query_sources
    nnstreamer_sources += join_paths(meson.current_source_dir(), s)
  endforeach
endif
