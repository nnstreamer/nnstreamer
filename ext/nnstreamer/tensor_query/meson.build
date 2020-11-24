tensor_query_common_deps = [
  nnstreamer_dep,
  glib_dep,
  gst_dep,
  edge_support_deps,
]

tensor_query_accept_source_files = [
  'tensor_query_accept.c',
]
tensor_query_accept_sources = []

foreach s : tensor_query_accept_source_files
  tensor_query_accept_sources += join_paths(meson.current_source_dir(), s)
endforeach

tensor_query_accept_dep = declare_dependency(
  sources : tensor_query_accept_sources,
  dependencies : tensor_query_common_deps,
)

tensor_query_reply_source_files = [
  'tensor_query_reply.c',
]
tensor_query_reply_sources = []

foreach s : tensor_query_reply_source_files
  tensor_query_reply_sources += join_paths(meson.current_source_dir(), s)
endforeach

tensor_query_reply_dep = declare_dependency(
  sources : tensor_query_reply_sources,
  dependencies : tensor_query_common_deps,
)

tensor_query_client_source_files = [
  'tensor_query_client.c',
]
tensor_query_client_sources = []

foreach s : tensor_query_client_source_files
  tensor_query_client_sources += join_paths(meson.current_source_dir(), s)
endforeach

tensor_query_client_dep = declare_dependency(
  sources : tensor_query_client_sources,
  dependencies : tensor_query_common_deps,
)
