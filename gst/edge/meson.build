edge_files = [
    'edge_common.c',
    'edge_elements.c',
    'edge_sink.c',
    'edge_src.c',
]

edge_srcs = []
edge_dep = [
    glib_dep, 
    gst_base_dep,
    gst_dep,
]

foreach s : edge_files
    edge_srcs += join_paths(meson.current_source_dir(), s)
endforeach

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
