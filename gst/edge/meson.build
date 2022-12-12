edge_srcs = [
    'edge_common.c',
    'edge_elements.c',
    'edge_sink.c',
    'edge_src.c',
]

edge_dep = [
    glib_dep,
    gst_base_dep,
    gst_dep,
    nnstreamer_edge_support_deps
]

if build_platform == 'tizen'
  edge_dep += dlog_dep
elif cc.has_header_symbol('android/log.h', '__android_log_print')
  edge_dep += cc.find_library('log')
endif

gstedge_shared = shared_library('gstedge',
  edge_srcs,
  dependencies: edge_dep,
  install: true,
  include_directories: include_directories('../nnstreamer/include'),
  install_dir: plugins_install_dir
)

gstedge_dep = declare_dependency(
    link_with: gstedge_shared,
    dependencies: edge_dep,
    include_directories: include_directories('.')
)
