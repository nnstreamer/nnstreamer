edge_srcs = [
    'edge_common.c',
    'edge_elements.c',
    'edge_sink.c',
    'edge_src.c',
]

edge_deps = [
    glib_dep,
    gst_base_dep,
    gst_dep,
    nnstreamer_dep,
    nnstreamer_edge_dep,
]

if build_platform == 'tizen'
  edge_deps += dlog_dep
elif cc.has_header_symbol('android/log.h', '__android_log_print')
  edge_deps += cc.find_library('log')
endif

gstedge_shared = shared_library('gstedge',
  edge_srcs,
  dependencies: edge_deps,
  install: true,
  install_dir: plugins_install_dir
)

gstedge_dep = declare_dependency(
    link_with: gstedge_shared,
    dependencies: edge_deps,
    include_directories: include_directories('.')
)
