join_files = [
  'gstjoin.c',
]

join_sources = []

foreach s : join_files
  join_sources += join_paths(meson.current_source_dir(), s)
endforeach

gstjoin_shared = shared_library('gstjoin',
  join_sources,
  dependencies: [glib_dep, gst_dep],
  install: true,
  install_dir: plugins_install_dir
)

gstjoin_static = static_library('gstjoin',
  join_sources,
  dependencies: [glib_dep, gst_dep],
  install: true,
  install_dir: nnstreamer_libdir
)

gstjoin_lib = gstjoin_shared
if get_option('default_library') == 'static'
  gstjoin_lib = gstjoin_static
endif

join_dep = declare_dependency(link_with: gstjoin_lib)
