join_sources = [
  'gstjoin.c',
]

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
