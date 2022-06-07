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

# nnstreamer-edge
# TODO:
# 1. This will be migrated into nnstreamer-edge repogitory.
# 2. glib dependency will be removed.
nnstreamer_edge_sources = [
  join_paths(meson.current_source_dir(), 'nnstreamer_edge_common.c')
]

nnstreamer_edge_deps = [
  glib_dep, gio_dep, thread_dep
]

if aitt_support_is_available
  nnstreamer_edge_sources += join_paths(meson.current_source_dir(), 'nnstreamer_edge_aitt.c')
  nnstreamer_edge_deps += aitt_support_deps
else
  nnstreamer_edge_sources += join_paths(meson.current_source_dir(), 'nnstreamer_edge_internal.c')
  if mqtt_support_is_available
    nnstreamer_edge_sources += join_paths(meson.current_source_dir(), 'nnstreamer_edge_mqtt.c')
    nnstreamer_edge_deps += [pahomqttc_dep, thread_dep]
  endif
endif

nns_edge_lib = shared_library('nnstreamer-edge',
  nnstreamer_edge_sources,
  dependencies: nnstreamer_edge_deps,
  install: true,
  install_dir: nnstreamer_libdir
)

static_library('nnstreamer-edge',
  nnstreamer_edge_sources,
  dependencies: nnstreamer_edge_deps,
  install: true,
  install_dir: nnstreamer_libdir
)

nns_edge_dep = declare_dependency(link_with: nns_edge_lib,
  dependencies: nnstreamer_edge_deps)
