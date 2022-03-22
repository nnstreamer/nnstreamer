tensor_query_sources = [
  'tensor_query_common.c',
  'tensor_query_serversrc.c',
  'tensor_query_serversink.c',
  'tensor_query_client.c',
  'tensor_query_server.c',
]

if query_hybrid_support_is_available
  tensor_query_sources += 'tensor_query_hybrid.c'
endif

foreach s : tensor_query_sources
  nnstreamer_sources += join_paths(meson.current_source_dir(), s)
endforeach

if aitt_support_is_available
# glib dependency will be removed.
  nnstreamer_edge_aitt_deps = aitt_support_deps + [glib_dep]

  nns_edge_aitt_sources = ['nnstreamer_edge_aitt.c']

  nnstreamer_edge_aitt_sources = []
  foreach s : nns_edge_aitt_sources
    nnstreamer_edge_aitt_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  nns_edge_aitt_lib = shared_library('nnstreamer_edge_aitt',
    nnstreamer_edge_aitt_sources,
    dependencies: nnstreamer_edge_aitt_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_edge_aitt',
    nnstreamer_edge_aitt_sources,
    dependencies: nnstreamer_edge_aitt_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )

  nns_edge_aitt_dep = declare_dependency(link_with: nns_edge_aitt_lib,
    dependencies: nnstreamer_edge_aitt_deps)
endif
